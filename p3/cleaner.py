"""LLM-based transcript cleaning and summarization."""

import asyncio
import inspect
import json
import os
import re
import threading
import random
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
import httpx

if TYPE_CHECKING:
    from .transcriber import AudioTranscriber

from .llm_context import (
    estimate_tokens,
    chunk_text,
    calculate_input_token_budget,
    DEFAULT_RESPONSE_TOKENS,
)

from .database import P3Database

# Optional Ollama support
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def load_api_key_from_file(file_path: str) -> Optional[str]:
    """Load API key from a file.

    Args:
        file_path: Path to the API key file (relative to project root or absolute)

    Returns:
        API key string or None if file not found or empty
    """
    try:
        # Convert to Path object for easier manipulation
        path = Path(file_path)

        # If relative path, try from project root
        if not path.is_absolute():
            # Assume project root is parent of p3 directory
            project_root = Path(__file__).parent.parent
            path = project_root / file_path

        if path.exists() and path.is_file():
            with open(path, 'r') as f:
                api_key = f.read().strip()
                if api_key and not api_key.startswith('your-') and not api_key.startswith('gsk_your_'):
                    return api_key
        return None
    except Exception as e:
        print(f"Warning: Could not load API key from {file_path}: {e}")
        return None


# Provider base URL mapping
PROVIDER_BASE_URLS = {
    'openrouter': 'https://openrouter.ai/api/v1',
    'groq': 'https://api.groq.com/openai/v1',
    'together': 'https://api.together.xyz/v1',
    'fireworks': 'https://api.fireworks.ai/inference/v1',
    'openai': 'https://api.openai.com/v1',
    'anthropic': 'https://api.anthropic.com/v1'
}

# Environment variable mapping
PROVIDER_ENV_VARS = {
    'openrouter': 'OPENROUTER_API_KEY',
    'groq': 'GROQ_API_KEY',
    'together': 'TOGETHER_API_KEY',
    'fireworks': 'FIREWORKS_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY'
}


class TranscriptCleaner:
    def __init__(self, db: P3Database, llm_provider: str = "openai",
                 llm_model: str = "gpt-3.5-turbo", api_key: str = None,
                 ollama_base_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3.2:3b",
                 api_key_file: str = None, base_url: str = None,
                 model_fallbacks: Optional[List[str]] = None,
                 timeout: int = 60, max_retries: int = 3,
                 transcriber: Optional["AudioTranscriber"] = None,
                 transcription_queue: Optional[Callable[[int], Optional[str]]] = None,
                 missing_transcript_status: str = "downloaded"):
        self.db = db
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model  # Local Ollama model for fallback
        self.model_fallbacks = model_fallbacks or []
        self.timeout = timeout
        self.max_retries = max_retries
        self.transcriber = transcriber
        self.transcription_queue = transcription_queue
        self.missing_transcript_status = missing_transcript_status

        # Determine base URL for external providers
        if base_url:
            self.base_url = base_url
        elif self.llm_provider in PROVIDER_BASE_URLS:
            self.base_url = PROVIDER_BASE_URLS[self.llm_provider]
        else:
            self.base_url = ollama_base_url

        # Load API key from file if specified
        if not self.api_key and api_key_file and self.llm_provider != "ollama":
            self.api_key = load_api_key_from_file(api_key_file)
            if self.api_key:
                print(f"Loaded API key from {api_key_file}")

        # Load API key from environment if not provided
        if not self.api_key and self.llm_provider != "ollama":
            env_var = PROVIDER_ENV_VARS.get(self.llm_provider)
            if env_var:
                self.api_key = os.getenv(env_var)
                if self.api_key:
                    print(f"Loaded API key from environment variable {env_var}")

        # Validate API key for non-Ollama providers
        if not self.api_key and self.llm_provider != "ollama":
            print(f"Warning: No API key found for {self.llm_provider}. "
                  f"Set {PROVIDER_ENV_VARS.get(self.llm_provider, 'API_KEY')} "
                  f"environment variable or provide api_key_file in config.")

    def _queue_transcription_job(self, episode_id: int) -> Optional[str]:
        if not self.transcription_queue:
            return None

        result = self.transcription_queue(episode_id)
        if inspect.isawaitable(result):
            try:
                return asyncio.run(result)
            except RuntimeError as exc:
                print(f"⚠ Unable to enqueue transcription job for episode {episode_id}: {exc}")
                return None
        return result

    def _handle_missing_transcript(self, episode_id: int) -> bool:
        self.db.update_episode_status(episode_id, self.missing_transcript_status)

        job_id = self._queue_transcription_job(episode_id)
        if job_id:
            print(f"↻ Queued transcription job {job_id} for episode {episode_id}")
            return False

        if self.transcriber:
            print(f"↻ Transcribing episode {episode_id} locally...")
            return self.transcriber.transcribe_episode(episode_id)

        print(f"⚠ No transcription queue or local transcriber available for episode {episode_id}")
        return False

    def clean_transcript(self, raw_text: str) -> str:
        """Clean transcript by removing filler words and improving readability."""
        # Basic cleaning without LLM first
        text = raw_text
        
        # Remove common filler words
        fillers = [
            r'\b(um|uh|ah|er|hmm|like|you know|sort of|kind of)\b',
            r'\b(actually|basically|literally|obviously|definitely)\b(?=\s+\w)',
        ]
        
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If we have API access or Ollama, use LLM for advanced cleaning
        if self.api_key or self.llm_provider == "ollama":
            try:
                text = self._llm_clean(text)
            except Exception as e:
                print(f"LLM cleaning failed, using basic cleaning: {e}")
        
        return text

    def _llm_clean(self, text: str) -> str:
        """Use LLM to clean and improve transcript.

        Falls back to local Ollama if cloud API fails with rate limit.
        """
        # OpenAI-compatible providers (OpenAI, OpenRouter, Groq, Together, Fireworks)
        if self.llm_provider in ["openai", "openrouter", "groq", "together", "fireworks"]:
            try:
                return self._openai_compatible_clean(text)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str or "413" in error_str:
                    print(
                        f"  ⚠ {self.llm_provider} API rate limited; "
                        "using basic cleanup instead."
                    )
                    return text
                raise
        elif self.llm_provider == "anthropic":
            return self._anthropic_clean(text)
        elif self.llm_provider == "ollama":
            return self._ollama_clean(text)
        else:
            print(f"Unsupported LLM provider: {self.llm_provider}")
            return text

    def _openai_compatible_clean(self, text: str) -> str:
        """Clean transcript using OpenAI-compatible API (OpenAI, OpenRouter, Groq, Together, Fireworks)."""
        prompt = """Clean this podcast transcript by:
1. Removing filler words (um, uh, like, you know)
2. Fixing grammar and punctuation
3. Preserving technical terms and proper nouns exactly
4. Maintaining the speaker's voice and meaning
5. Breaking into clear paragraphs

Return only the cleaned text, no additional commentary.

Transcript:
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # OpenRouter requires additional headers
        if self.llm_provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/your-repo/parakeet-podcast-processor"
            headers["X-Title"] = "Parakeet Podcast Processor"

        model_candidates = [self.llm_model] + [m for m in self.model_fallbacks if m != self.llm_model]
        last_error = None

        for model_index, model_name in enumerate(model_candidates):
            response = None
            size_error = False
            max_retries = max(1, self.max_retries)
            for attempt in range(max_retries):
                with httpx.Client(timeout=float(self.timeout)) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": "You are an expert transcript editor."},
                                {"role": "user", "content": prompt + text}
                            ],
                            "temperature": 0.1,
                            "max_tokens": min(len(text) * 2, 4000)
                        }
                    )

                if self._is_request_too_large(response):
                    print(
                        f"  ⚠ {self.llm_provider} request too large for {model_name}. "
                        "Trying next model fallback..."
                    )
                    size_error = True
                    break
                if not self._is_rate_limit_response(response):
                    break

                wait_seconds = self._parse_retry_seconds(response)
                if wait_seconds is None:
                    wait_seconds = min(3 ** attempt + 2, 30)
                wait_seconds += random.uniform(0.5, 2.0)
                if attempt < max_retries - 1:
                    print(
                        f"  ⚠ {self.llm_provider} rate limited, retrying after "
                        f"{wait_seconds:.1f}s..."
                    )
                    time.sleep(wait_seconds)
                    continue
                print(
                    f"  ⚠ {self.llm_provider} rate limited after {max_retries} attempts."
                )

            if size_error:
                continue
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()

            if self._is_rate_limit_response(response):
                last_error = response
                if model_index < len(model_candidates) - 1:
                    print(
                        f"  ⚠ {self.llm_provider} still rate limited after retries. "
                        "Trying next model fallback..."
                    )
                continue

            raise Exception(f"{self.llm_provider} API error: {response.status_code} - {response.text}")

        if last_error is not None:
            raise Exception(f"{self.llm_provider} API error: {last_error.status_code} - {last_error.text}")
        raise Exception(f"{self.llm_provider} API error: no response")

    def _anthropic_clean(self, text: str) -> str:
        """Clean transcript using Anthropic Claude API."""
        # Similar implementation for Claude
        # This would use the Anthropic API format
        return text  # Placeholder

    def _ollama_clean(self, text: str) -> str:
        """Clean transcript using Ollama local LLM."""
        if not OLLAMA_AVAILABLE:
            print("Ollama not available, skipping LLM cleaning")
            return text
            
        prompt = """Clean this podcast transcript by:
1. Removing filler words (um, uh, like, you know)
2. Fixing grammar and punctuation
3. Preserving technical terms and proper nouns exactly
4. Maintaining the speaker's voice and meaning
5. Breaking into clear paragraphs

Return only the cleaned text, no additional commentary.

Transcript:
""" + text

        try:
            response = ollama.chat(
                model=self.ollama_model,  # Use local Ollama model name, not cloud API model
                messages=[
                    {"role": "system", "content": "You are an expert transcript editor."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Ollama cleaning failed: {e}")
            return text

    def generate_summary(self, episode_id: int) -> Dict[str, Any]:
        """Generate structured summary of an episode."""
        # Get transcript segments
        segments = self.db.get_transcripts_for_episode(episode_id)
        full_text = "\n".join(segment['text'] for segment in segments)
        
        if not full_text.strip():
            return None
        
        # Clean the transcript first
        cleaned_text = self.clean_transcript(full_text)
        
        # Generate structured summary using LLM
        summary_data = self._generate_structured_summary(cleaned_text)
        
        if summary_data:
            # Store in database
            self.db.add_summary(
                episode_id=episode_id,
                key_topics=summary_data.get('key_topics', []),
                themes=summary_data.get('themes', []),
                quotes=summary_data.get('quotes', []),
                startups=summary_data.get('startups', []),
                full_summary=summary_data.get('summary', ''),
                digest_date=datetime.now(),
                key_insights=summary_data.get('key_insights', []),
                actionable_takeaways=summary_data.get('actionable_takeaways', [])
            )
            
            # Update episode status
            self.db.update_episode_status(episode_id, 'processed')
        
        return summary_data

    def _merge_summary_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge summary results from multiple chunks into a single result.

        Args:
            results: List of summary dictionaries from each chunk

        Returns:
            Merged summary dictionary with deduplicated lists and combined text
        """
        if not results:
            return {}

        if len(results) == 1:
            return results[0]

        merged = {
            'key_topics': [],
            'themes': [],
            'quotes': [],
            'startups': [],
            'key_insights': [],
            'actionable_takeaways': [],
            'summary': ''
        }

        # Collect all items from all chunks
        for result in results:
            if result:
                merged['key_topics'].extend(result.get('key_topics', []))
                merged['themes'].extend(result.get('themes', []))
                merged['quotes'].extend(result.get('quotes', []))
                merged['startups'].extend(result.get('startups', []))
                merged['key_insights'].extend(result.get('key_insights', []))
                merged['actionable_takeaways'].extend(result.get('actionable_takeaways', []))

        # Deduplicate while preserving order
        def dedupe(lst):
            seen = set()
            result = []
            for item in lst:
                item_lower = item.lower().strip() if isinstance(item, str) else str(item)
                if item_lower not in seen:
                    seen.add(item_lower)
                    result.append(item)
            return result

        merged['key_topics'] = dedupe(merged['key_topics'])[:7]  # Limit to 7 topics
        merged['themes'] = dedupe(merged['themes'])[:5]  # Limit to 5 themes
        merged['quotes'] = dedupe(merged['quotes'])[:8]  # Limit to 8 quotes
        merged['startups'] = dedupe(merged['startups'])
        merged['key_insights'] = dedupe(merged['key_insights'])[:6]  # Limit to 6 insights
        merged['actionable_takeaways'] = dedupe(merged['actionable_takeaways'])[:5]  # Limit to 5 takeaways

        # Combine summaries from all chunks
        summaries = [r.get('summary', '') for r in results if r and r.get('summary')]
        if summaries:
            # Join chunk summaries and note it's a multi-part summary
            merged['summary'] = ' '.join(summaries)

        return merged

    def _generate_structured_summary(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate structured summary using LLM.

        Handles large transcripts by chunking them to stay within API token limits.
        """
        if not self.api_key and self.llm_provider != "ollama":
            # Fallback to simple extraction
            return self._basic_extraction(text)

        prompt = """Analyze this podcast transcript and extract structured information in JSON format, optimized for tech investing insights:

{
  "key_topics": ["topic1", "topic2", ...],
  "themes": ["theme1", "theme2", ...],
  "quotes": ["notable quote 1", "notable quote 2", ...],
  "startups": ["company1", "company2", ...],
  "key_insights": ["insight1", "insight2", ...],
  "actionable_takeaways": ["takeaway1", "takeaway2", ...],
  "summary": "Detailed 4-6 sentence summary of the episode"
}

Guidelines for Tech Investing Context:
- key_topics: Main subjects discussed - prioritize business models, market trends, competitive dynamics, technology shifts (4-7 topics)
- themes: Broader investment themes (e.g., "AI infrastructure buildout", "vertical SaaS adoption", "consumer behavior shifts") (3-5 themes)
- quotes: CRITICAL - Select quotes that provide tactical/strategic value for investors:
  * Specific numbers (revenue, growth rates, market size, valuations, unit economics)
  * Strategic insights (competitive moats, go-to-market strategies, product differentiation)
  * Market observations (trend predictions, industry analysis, customer behavior)
  * Investor perspectives (what makes companies succeed/fail, investment frameworks)
  * NOT generic inspiration or obvious advice - focus on substantive, non-obvious insights
  * Include 5-8 of the most valuable quotes only
- startups: Any companies, startups, or brands mentioned (include both private and public companies)
- key_insights: Investment-relevant insights such as:
  * Business model analysis and scalability factors
  * Competitive landscape and market positioning
  * Growth strategies and distribution channels
  * Unit economics, margins, and path to profitability
  * Technology trends and their business implications
  * Customer acquisition and retention dynamics
  * (4-6 insights that help understand companies/markets better)
- actionable_takeaways: Practical knowledge for tech investing:
  * What to look for when evaluating similar companies
  * Key questions to ask founders/management
  * Red flags or green flags to watch for
  * Market dynamics to monitor
  * (3-5 investor-focused takeaways)
- summary: Comprehensive 4-6 sentence overview focusing on business/market insights, key arguments about companies/sectors, and investment-relevant conclusions

Focus on substance over style. Prioritize information that helps understand businesses, markets, and investment opportunities.

Transcript:
"""

        max_input_tokens = calculate_input_token_budget(
            self.llm_model,
            prompt,
            desired_response_tokens=DEFAULT_RESPONSE_TOKENS,
        )
        estimated_tokens = estimate_tokens(text)

        if estimated_tokens > max_input_tokens:
            print(
                f"  📄 Large transcript ({estimated_tokens} est. tokens), "
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
            else:
                return self._basic_extraction(text)
        
        try:
            # OpenAI-compatible providers (OpenAI, OpenRouter, Groq, Together, Fireworks)
            if self.llm_provider in ["openai", "openrouter", "groq", "together", "fireworks"]:
                try:
                    result = self._openai_compatible_extract(prompt + text)
                    return result or self._basic_extraction(text)
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "rate_limit" in error_str or "413" in error_str:
                        print(
                            f"  ⚠ {self.llm_provider} API rate limited; "
                            "using basic extraction instead of Ollama."
                        )
                        return self._basic_extraction(text)
                    raise
            elif self.llm_provider == "anthropic":
                return self._anthropic_extract(prompt + text)
            elif self.llm_provider == "ollama":
                result = self._ollama_extract(prompt + text)
                return result or self._basic_extraction(text)
        except Exception as e:
            print(f"LLM summarization failed: {e}")
            return self._basic_extraction(text)

    def _process_single_chunk(self, chunk: str, chunk_num: int, total_chunks: int) -> Optional[Dict[str, Any]]:
        """Process a single chunk of transcript for summarization.

        Args:
            chunk: Text chunk to process
            chunk_num: Current chunk number (1-indexed)
            total_chunks: Total number of chunks

        Returns:
            Summary dict for this chunk or None if failed
        """
        # Modify prompt to indicate partial transcript
        chunk_context = f"This is part {chunk_num} of {total_chunks} of a podcast transcript. " if total_chunks > 1 else ""

        prompt = f"""{chunk_context}Analyze this podcast transcript section and extract structured information in JSON format, optimized for tech investing insights:

{{
  "key_topics": ["topic1", "topic2", ...],
  "themes": ["theme1", "theme2", ...],
  "quotes": ["notable quote 1", "notable quote 2", ...],
  "startups": ["company1", "company2", ...],
  "key_insights": ["insight1", "insight2", ...],
  "actionable_takeaways": ["takeaway1", "takeaway2", ...],
  "summary": "2-3 sentence summary of this section"
}}

Guidelines:
- key_topics: Main subjects discussed in this section (2-4 topics)
- themes: Investment themes (e.g., "AI infrastructure", "vertical SaaS") (1-3 themes)
- quotes: Select 2-4 quotes with tactical/strategic value (specific numbers, strategic insights, market observations)
- startups: Companies or brands mentioned
- key_insights: Investment-relevant insights (2-3 insights)
- actionable_takeaways: Practical knowledge (1-2 takeaways)
- summary: Brief 2-3 sentence overview of this section

Transcript section:
"""

        try:
            # OpenAI-compatible providers
            if self.llm_provider in ["openai", "openrouter", "groq", "together", "fireworks"]:
                try:
                    result = self._openai_compatible_extract(prompt + chunk)
                    return result or self._basic_extraction(chunk)
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "rate_limit" in error_str or "413" in error_str:
                        print(
                            f"    ⚠ {self.llm_provider} API rate limited on chunk {chunk_num}; "
                            "using basic extraction instead of Ollama."
                        )
                        return self._basic_extraction(chunk)
                    raise
            elif self.llm_provider == "anthropic":
                return self._anthropic_extract(prompt + chunk)
            elif self.llm_provider == "ollama":
                result = self._ollama_extract(prompt + chunk)
                return result or self._basic_extraction(chunk)
        except Exception as e:
            print(f"    ⚠ Failed to process chunk {chunk_num}: {e}")
            return None

        return None

    def _openai_compatible_extract(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Extract using OpenAI-compatible API (OpenAI, OpenRouter, Groq, Together, Fireworks)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # OpenRouter requires additional headers
        if self.llm_provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/your-repo/parakeet-podcast-processor"
            headers["X-Title"] = "Parakeet Podcast Processor"

        # Some providers support response_format for JSON mode
        json_payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": "You are an expert at analyzing podcast content. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }

        # OpenAI and some providers support JSON mode
        if self.llm_provider in ["openai", "openrouter"]:
            json_payload["response_format"] = {"type": "json_object"}

        model_candidates = [self.llm_model] + [m for m in self.model_fallbacks if m != self.llm_model]

        for model_index, model_name in enumerate(model_candidates):
            response = None
            json_payload["model"] = model_name
            size_error = False
            max_retries = max(1, self.max_retries)
            for attempt in range(max_retries):
                with httpx.Client(timeout=float(self.timeout * 2)) as client:  # Double timeout for extraction
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=json_payload
                    )

                if self._is_request_too_large(response):
                    print(
                        f"  ⚠ {self.llm_provider} request too large for {model_name}. "
                        "Trying next model fallback..."
                    )
                    size_error = True
                    break
                if not self._is_rate_limit_response(response):
                    break

                wait_seconds = self._parse_retry_seconds(response)
                if wait_seconds is None:
                    wait_seconds = min(3 ** attempt + 2, 30)
                wait_seconds += random.uniform(0.5, 2.0)
                if attempt < max_retries - 1:
                    print(
                        f"  ⚠ {self.llm_provider} rate limited, retrying after "
                        f"{wait_seconds:.1f}s..."
                    )
                    time.sleep(wait_seconds)
                    continue
                print(
                    f"  ⚠ {self.llm_provider} rate limited after {max_retries} attempts."
                )

            if size_error:
                continue
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                # Extract JSON from response (in case there's extra text)
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)

                    # Ensure all expected fields exist with defaults
                    parsed.setdefault('key_topics', [])
                    parsed.setdefault('themes', [])
                    parsed.setdefault('quotes', [])
                    parsed.setdefault('startups', [])
                    parsed.setdefault('key_insights', [])
                    parsed.setdefault('actionable_takeaways', [])
                    parsed.setdefault('summary', '')

                    return parsed

            if not self._is_rate_limit_response(response):
                return None
            if model_index < len(model_candidates) - 1:
                print(
                    f"  ⚠ {self.llm_provider} still rate limited after retries. "
                    "Trying next model fallback..."
                )

        return None

    def _parse_retry_seconds(self, response: httpx.Response) -> Optional[int]:
        """Parse retry seconds from API rate limit responses."""
        retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

        try:
            message = response.json().get("error", {}).get("message", "")
        except (ValueError, AttributeError, TypeError):
            message = response.text

        match = re.search(r"try again in (\\d+)s", message or "", re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    @staticmethod
    def _is_rate_limit_response(response: httpx.Response) -> bool:
        if response.status_code == 429:
            return True
        try:
            error = response.json().get("error", {})
        except (ValueError, AttributeError, TypeError):
            return False
        code = str(error.get("code", "")).lower()
        message = str(error.get("message", "")).lower()
        if "rate_limit" in code:
            return True
        return "tokens per minute" in message or "rate limit" in message

    @staticmethod
    def _is_request_too_large(response: httpx.Response) -> bool:
        try:
            message = response.json().get("error", {}).get("message", "")
        except (ValueError, AttributeError, TypeError):
            return False
        lowered = message.lower()
        return (
            "request too large" in lowered
            or "context length" in lowered
            or "maximum context" in lowered
        )

    def _anthropic_extract(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Extract using Anthropic Claude API."""
        # Placeholder for Claude implementation
        return None

    def _ollama_extract(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Extract structured data using Ollama."""
        if not OLLAMA_AVAILABLE:
            print("Ollama not available, using basic extraction")
            return None

        try:
            print(f"Generating summary with {self.ollama_model}...", flush=True)
            stop_event = threading.Event()

            def spinner():
                while not stop_event.wait(1.0):
                    print(".", end="", flush=True)

            spinner_thread = threading.Thread(target=spinner, daemon=True)
            spinner_thread.start()

            try:
                # Stream the response to show progress
                # Use format='json' to enforce valid JSON output
                response = ollama.chat(
                    model=self.ollama_model,  # Use local Ollama model name, not cloud API model
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing podcast content. Return valid JSON only with all requested fields."},
                        {"role": "user", "content": prompt}
                    ],
                    format='json',  # Enforce JSON output format
                    stream=True,
                    options={
                        "temperature": 0.3,  # Lower temperature for more consistent output
                        "num_predict": 2000   # Allow longer responses for detailed summaries
                    }
                )

                # Collect streamed response
                full_content = ""
                for chunk in response:
                    if chunk['message']['content']:
                        full_content += chunk['message']['content']
            finally:
                stop_event.set()
                spinner_thread.join()
                print(" done!", flush=True)

            content = full_content.strip()

            # With format='json', the output should already be valid JSON
            # But we still extract it in case there's any wrapping
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed = json.loads(json_str)

                # Ensure all expected fields exist with defaults
                parsed.setdefault('key_topics', [])
                parsed.setdefault('themes', [])
                parsed.setdefault('quotes', [])
                parsed.setdefault('startups', [])
                parsed.setdefault('key_insights', [])
                parsed.setdefault('actionable_takeaways', [])
                parsed.setdefault('summary', '')

                return parsed

            return None
        except json.JSONDecodeError as e:
            print(f"Ollama extraction failed: {e}")
            # Try to salvage partial data if possible
            return None
        except Exception as e:
            print(f"Ollama extraction failed: {e}")
            return None

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
        
        # Simple company extraction (words ending in common suffixes)
        potential_companies = []
        for word in text.split():
            if any(word.lower().endswith(suffix) for suffix in ['inc', 'corp', 'llc', 'labs']):
                potential_companies.append(word)
        
        return {
            "key_topics": key_topics,
            "themes": ["general discussion"],
            "quotes": [],
            "startups": list(set(potential_companies)),
            "key_insights": [],
            "actionable_takeaways": [],
            "summary": "Podcast episode discussion covering various topics."
        }

    def process_all_transcribed(self) -> int:
        """Process all episodes with 'transcribed' status."""
        episodes = self.db.get_episodes_by_status('transcribed')
        processed_count = 0
        
        for episode in episodes:
            print(f"Processing summary for: {episode['title']}")

            transcripts = self.db.get_transcripts_for_episode(episode['id'])
            if not transcripts:
                if not episode.get("file_path") or not os.path.exists(episode["file_path"]):
                    print(f"✗ Missing audio file for episode {episode['id']}: {episode.get('file_path')}")
                print(f"✗ Missing transcripts for episode {episode['id']}")
                transcribed_now = self._handle_missing_transcript(episode['id'])
                if transcribed_now and self.generate_summary(episode['id']):
                    processed_count += 1
                    print(f"✓ Processed: {episode['title']}")
                continue

            if not episode.get("file_path") or not os.path.exists(episode["file_path"]):
                print(
                    f"⚠ Missing audio file for episode {episode['id']}: "
                    f"{episode.get('file_path')} (continuing with transcript)"
                )

            if self.generate_summary(episode['id']):
                processed_count += 1
                print(f"✓ Processed: {episode['title']}")
            else:
                print(f"✗ Failed to process: {episode['title']}")
        
        return processed_count

