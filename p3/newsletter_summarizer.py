"""Newsletter summarization using LLMs."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .database import P3Database
from .llm_client import LLMClient
from .llm_context import (
    estimate_tokens,
    chunk_text,
    calculate_input_token_budget,
    DEFAULT_RESPONSE_TOKENS,
)


class NewsletterSummarizer:
    def __init__(self, db: P3Database, llm_provider: str = "ollama",
                 llm_model: str = "llama3.2:1b", api_key: str = None,
                 ollama_base_url: str = "http://localhost:11434",
                 api_key_file: str = None, base_url: str = None,
                 timeout: int = 60, max_retries: int = 3,
                 model_fallbacks: List[str] = None,
                 llm_client: LLMClient = None):
        self.db = db
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model
        self.timeout = timeout
        self.max_retries = max_retries

        # Use provided client or create one
        self.llm_client = llm_client or LLMClient(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key,
            api_key_file=api_key_file,
            base_url=base_url,
            model_fallbacks=model_fallbacks or [],
            ollama_model=llm_model,
            ollama_base_url=ollama_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate_summary(self, article_id: int) -> Dict[str, Any]:
        """Generate structured summary of a newsletter article."""
        # Get article content
        article = self.db.conn.execute("""
            SELECT id, newsletter_id, title, content, url
            FROM newsletter_articles
            WHERE id = ?
        """, (article_id,)).fetchone()

        if not article or not article[3]:  # No content
            return None

        content = article[3]
        if not content.strip():
            return None

        # Generate structured summary using LLM
        summary_data = self._generate_structured_summary(content)

        if summary_data:
            # Store in database
            self.db.add_newsletter_summary(
                article_id=article_id,
                key_topics=summary_data.get('key_topics', []),
                themes=summary_data.get('themes', []),
                quotes=summary_data.get('quotes', []),
                companies=summary_data.get('companies', []),
                full_summary=summary_data.get('summary', ''),
                digest_date=datetime.now(),
                key_insights=summary_data.get('key_insights', []),
                actionable_takeaways=summary_data.get('actionable_takeaways', [])
            )

            # Update article status
            self.db.update_article_status(article_id, 'processed')

        return summary_data

    def _merge_summary_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge summary results from multiple chunks into a single result."""
        if not results:
            return {}

        if len(results) == 1:
            return results[0]

        merged = {
            "key_topics": [],
            "themes": [],
            "quotes": [],
            "companies": [],
            "key_insights": [],
            "actionable_takeaways": [],
            "summary": "",
        }

        for result in results:
            if result:
                merged["key_topics"].extend(result.get("key_topics", []))
                merged["themes"].extend(result.get("themes", []))
                merged["quotes"].extend(result.get("quotes", []))
                merged["companies"].extend(result.get("companies", []))
                merged["key_insights"].extend(result.get("key_insights", []))
                merged["actionable_takeaways"].extend(result.get("actionable_takeaways", []))

        def dedupe(items):
            seen = set()
            output = []
            for item in items:
                item_key = item.lower().strip() if isinstance(item, str) else str(item)
                if item_key not in seen:
                    seen.add(item_key)
                    output.append(item)
            return output

        merged["key_topics"] = dedupe(merged["key_topics"])[:7]
        merged["themes"] = dedupe(merged["themes"])[:5]
        merged["quotes"] = dedupe(merged["quotes"])[:8]
        merged["companies"] = dedupe(merged["companies"])
        merged["key_insights"] = dedupe(merged["key_insights"])[:6]
        merged["actionable_takeaways"] = dedupe(merged["actionable_takeaways"])[:5]

        summaries = [r.get("summary", "") for r in results if r and r.get("summary")]
        if summaries:
            merged["summary"] = " ".join(summaries)

        return merged

    def _process_single_chunk(
        self,
        chunk: str,
        chunk_num: int,
        total_chunks: int,
    ) -> Optional[Dict[str, Any]]:
        """Process a single chunk of a newsletter article for summarization."""
        chunk_context = f"This is part {chunk_num} of {total_chunks} of a newsletter article. " if total_chunks > 1 else ""

        prompt = f"""{chunk_context}Analyze this newsletter article section and extract structured information in JSON format, optimized for tech investing insights:

{{
  "key_topics": ["topic1", "topic2", ...],
  "themes": ["theme1", "theme2", ...],
  "quotes": ["notable quote 1", "notable quote 2", ...],
  "companies": ["company1", "company2", ...],
  "key_insights": ["insight1", "insight2", ...],
  "actionable_takeaways": ["takeaway1", "takeaway2", ...],
  "summary": "2-3 sentence summary of this section"
}}

Guidelines:
- key_topics: Main subjects discussed in this section (2-4 topics)
- themes: Investment themes (1-3 themes)
- quotes: Select 2-4 quotes with tactical/strategic value (specific numbers, strategic insights, market observations)
- companies: Companies or brands mentioned
- key_insights: Investment-relevant insights (2-3 insights)
- actionable_takeaways: Practical knowledge (1-2 takeaways)
- summary: Brief 2-3 sentence overview of this section

Article section:
"""

        try:
            result = self._extract_with_llm(prompt + chunk)
            return result or self._basic_extraction(chunk)
        except Exception as e:
            print(f"    ⚠ Failed to process chunk {chunk_num}: {e}")
            return None

    def _generate_structured_summary(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate structured summary using LLM."""
        prompt = """Analyze this newsletter article and extract structured information in JSON format, optimized for tech investing insights:

{
  "key_topics": ["topic1", "topic2", ...],
  "themes": ["theme1", "theme2", ...],
  "quotes": ["notable quote 1", "notable quote 2", ...],
  "companies": ["company1", "company2", ...],
  "key_insights": ["insight1", "insight2", ...],
  "actionable_takeaways": ["takeaway1", "takeaway2", ...],
  "summary": "Detailed 4-6 sentence summary of the article"
}

Guidelines for Tech Investing Context:
- key_topics: Main subjects discussed - prioritize business models, market trends, competitive dynamics, technology shifts (4-7 topics)
- themes: Broader investment themes (e.g., "AI infrastructure buildout", "vertical SaaS adoption", "consumer behavior shifts") (3-5 themes)
- quotes: CRITICAL - Select quotes that provide tactical/strategic value for investors:
  * Specific numbers (revenue, growth rates, market size, valuations, unit economics)
  * Strategic insights (competitive moats, go-to-market strategies, product differentiation)
  * Market observations (trend predictions, industry analysis, customer behavior)
  * Investor perspectives (what makes companies succeed/fail, investment frameworks)
  * Technical analysis and expert commentary
  * NOT generic inspiration or obvious advice - focus on substantive, non-obvious insights
  * Include 5-8 of the most valuable quotes only
- companies: Any companies, startups, or brands mentioned (include both private and public companies)
- key_insights: Investment-relevant insights such as:
  * Business model analysis and scalability factors
  * Competitive landscape and market positioning
  * Growth strategies and distribution channels
  * Unit economics, margins, and path to profitability
  * Technology trends and their business implications
  * Customer acquisition and retention dynamics
  * Industry structure and power dynamics
  * (4-6 insights that help understand companies/markets better)
- actionable_takeaways: Practical knowledge for tech investing:
  * What to look for when evaluating similar companies
  * Key questions to ask founders/management
  * Red flags or green flags to watch for
  * Market dynamics to monitor
  * Investment frameworks or mental models
  * (3-5 investor-focused takeaways)
- summary: Comprehensive 4-6 sentence overview focusing on business/market insights, key arguments about companies/sectors, and investment-relevant conclusions

Focus on substance over style. Prioritize information that helps understand businesses, markets, and investment opportunities.

Article:
"""

        max_input_tokens = calculate_input_token_budget(
            self.llm_model,
            prompt,
            desired_response_tokens=DEFAULT_RESPONSE_TOKENS,
        )
        estimated_tokens = estimate_tokens(text)

        if estimated_tokens > max_input_tokens:
            print(
                f"  📄 Large article ({estimated_tokens} est. tokens), "
                f"processing in chunks (budget {max_input_tokens} tokens)..."
            )
            overlap_tokens = min(200, max(50, max_input_tokens // 10))
            chunks = chunk_text(text, max_tokens=max_input_tokens, overlap_tokens=overlap_tokens)
            print(f"  📄 Split into {len(chunks)} chunks")

            chunk_results = []
            for i, chunk in enumerate(chunks):
                print(f"  📄 Processing chunk {i+1}/{len(chunks)}...")
                chunk_result = self._process_single_chunk(chunk, i + 1, len(chunks))
                if chunk_result:
                    chunk_results.append(chunk_result)

            if chunk_results:
                return self._merge_summary_results(chunk_results)
            return self._basic_extraction(text)

        try:
            result = self._extract_with_llm(prompt + text)
            return result or self._basic_extraction(text)
        except Exception as e:
            print(f"LLM summarization failed: {e}")
            return self._basic_extraction(text)

    def _extract_with_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Extract structured data using the shared LLM client with retry/fallback."""
        result = self.llm_client.chat_json(
            messages=[
                {"role": "system", "content": "You are an expert at analyzing tech/business content. Return valid JSON only with all requested fields."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )

        if result:
            # Ensure all expected fields exist
            result.setdefault('key_topics', [])
            result.setdefault('themes', [])
            result.setdefault('quotes', [])
            result.setdefault('companies', [])
            result.setdefault('key_insights', [])
            result.setdefault('actionable_takeaways', [])
            result.setdefault('summary', '')

        return result

    def _basic_extraction(self, text: str) -> Dict[str, Any]:
        """Basic keyword extraction as fallback."""
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get most frequent words as topics
        key_topics = [word for word, freq in
                     sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]]

        return {
            "key_topics": key_topics,
            "themes": ["general discussion"],
            "quotes": [],
            "companies": [],
            "key_insights": [],
            "actionable_takeaways": [],
            "summary": "Newsletter article covering various topics."
        }

    def process_all_fetched(self) -> int:
        """Process all articles with 'fetched' status."""
        articles = self.db.get_articles_by_status('fetched')
        processed_count = 0

        for article in articles:
            print(f"Processing summary for: {article['title']}")
            if self.generate_summary(article['id']):
                processed_count += 1
                print(f"✓ Processed: {article['title']}")
            else:
                print(f"✗ Failed to process: {article['title']}")

        return processed_count

