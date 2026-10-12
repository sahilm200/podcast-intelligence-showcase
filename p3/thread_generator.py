"""Twitter thread generation from podcast digests."""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from .llm_client import LLMClient


class ThreadGenerator:
    def __init__(self, llm_model: str = "llama3.2:latest",
                 llm_client: LLMClient = None):
        self.llm_model = llm_model
        self.llm_client = llm_client or LLMClient(
            provider="ollama", model=llm_model
        )

    def generate_thread(self, summaries: List[Dict[str, Any]], topic: str,
                       style: str = "educational", max_tweets: int = 10,
                       extra_instructions: str = None) -> Dict[str, Any]:
        """Generate Twitter thread from podcast summaries.

        Args:
            summaries: List of podcast summary dicts
            topic: Thread topic/angle
            style: Thread style (educational, contrarian, hot-take, data-driven)
            max_tweets: Maximum number of tweets (default 10)
            extra_instructions: Optional additional instructions for tone, structure, or focus

        Returns:
            Dict with thread tweets, metadata, and copy-paste ready version
        """
        if not self.llm_client:
            return {"error": "No LLM client available"}

        # Combine insights from summaries
        context = self._build_thread_context(summaries, topic)

        # Generate thread based on style
        if style == "educational":
            prompt = self._educational_prompt(topic, context, max_tweets)
        elif style == "contrarian":
            prompt = self._contrarian_prompt(topic, context, max_tweets)
        elif style == "hot-take":
            prompt = self._hot_take_prompt(topic, context, max_tweets)
        elif style == "data-driven":
            prompt = self._data_driven_prompt(topic, context, max_tweets)
        else:
            prompt = self._educational_prompt(topic, context, max_tweets)

        # Append extra instructions if provided
        if extra_instructions:
            prompt += f"""

Additional Instructions from the author:
{extra_instructions}
"""

        # Generate thread
        raw_thread = self._generate_with_llm(prompt)

        # Parse into individual tweets
        tweets = self._parse_thread(raw_thread)

        # Validate tweet lengths
        tweets = self._validate_tweets(tweets)

        # Create copy-paste version
        copy_paste = self._format_for_twitter(tweets)

        return {
            'tweets': tweets,
            'tweet_count': len(tweets),
            'style': style,
            'topic': topic,
            'source_count': len(summaries),
            'copy_paste': copy_paste,
            'mode': 'thread',
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'model': self.llm_model
            }
        }

    def generate_single_tweet(self, summaries: List[Dict[str, Any]], topic: str,
                              extra_instructions: str = None) -> Dict[str, Any]:
        """Generate a single standalone tweet that distills a key insight.

        Args:
            summaries: List of podcast summary dicts
            topic: Tweet topic/angle
            extra_instructions: Optional additional instructions

        Returns:
            Dict with tweet text, metadata
        """
        if not self.llm_client:
            return {"error": "No LLM client available"}

        # Use compact context for single tweet - less tokens needed than a thread
        context = self._build_tweet_context(summaries, topic)

        prompt = f"""Write ONE standalone tweet about "{topic}" based on podcast insights.

{context}

Requirements:
- MUST be under 280 characters
- Distill the single most valuable, unique, or surprising insight
- Make it self-contained - no "thread" or "1/" numbering
- Use a strong hook: a bold claim, surprising stat, or contrarian take
- Be specific, not generic - reference concrete details from the sources
- Write in a voice that sounds like a smart insider sharing alpha
- Do NOT use emojis or hashtags

Return ONLY the tweet text, nothing else."""

        if extra_instructions:
            prompt += f"""

Additional Instructions from the author:
{extra_instructions}
"""

        raw_tweet = self._generate_with_llm(prompt)

        # Clean up - strip quotes and whitespace
        tweet = raw_tweet.strip().strip('"').strip("'").strip()

        # Truncate if needed
        if len(tweet) > 280:
            tweet = tweet[:277] + '...'

        return {
            'tweet': tweet,
            'char_count': len(tweet),
            'topic': topic,
            'source_count': len(summaries),
            'mode': 'single',
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'model': self.llm_model
            }
        }

    def _build_thread_context(self, summaries: List[Dict[str, Any]], topic: str) -> str:
        """Build context from summaries."""
        all_insights = []
        all_quotes = []
        all_takeaways = []
        all_companies = []
        sources = []

        for s in summaries:
            source_label = f"{s.get('podcast_title', 'Unknown')} - {s.get('episode_title', 'Unknown')}"
            sources.append(source_label)
            all_insights.extend(s.get('key_insights', []))
            # Attribute quotes to their source episode
            for q in s.get('quotes', []):
                all_quotes.append(f"{q} -- {source_label}")
            all_takeaways.extend(s.get('actionable_takeaways', []))
            all_companies.extend(s.get('startups', []))

        context = f"""
Topic: {topic}

Sources ({len(summaries)} episodes):
{chr(10).join('- ' + src for src in sources[:5])}

Key Insights:
{chr(10).join('- ' + insight for insight in all_insights[:10])}

Notable Quotes (with source):
{chr(10).join('- "' + quote + '"' for quote in all_quotes[:8])}

Actionable Takeaways:
{chr(10).join('- ' + takeaway for takeaway in all_takeaways[:6])}

Companies Mentioned:
{', '.join(set(all_companies[:15]))}
"""
        return context

    def _build_tweet_context(self, summaries: List[Dict[str, Any]], topic: str) -> str:
        """Build compact context for a single tweet - fewer tokens than thread context."""
        all_insights = []
        all_takeaways = []
        sources = []

        for s in summaries:
            sources.append(f"{s.get('podcast_title', 'Unknown')} - {s.get('episode_title', 'Unknown')}")
            all_insights.extend(s.get('key_insights', []))
            all_takeaways.extend(s.get('actionable_takeaways', []))

        return f"""Topic: {topic}

Sources ({len(summaries)} episodes):
{chr(10).join('- ' + src for src in sources[:3])}

Key Insights:
{chr(10).join('- ' + insight for insight in all_insights[:5])}

Takeaways:
{chr(10).join('- ' + t for t in all_takeaways[:3])}
"""

    def _educational_prompt(self, topic: str, context: str, max_tweets: int) -> str:
        """Educational thread prompt."""
        return f"""Generate an educational Twitter thread about "{topic}" based on podcast insights.

{context}

Requirements:
- {max_tweets} tweets maximum
- Tweet 1: Strong hook that makes people want to read more (ask a question, make a bold claim, share surprising stat)
- Tweets 2-{max_tweets-2}: Numbered insights (1/, 2/, 3/, etc.) - each tweet is self-contained
- Tweet {max_tweets-1}: Key takeaway or action item
- Tweet {max_tweets}: Call-to-action (ask readers to share, follow, engage)
- Each tweet MUST be under 280 characters
- Include relevant quotes from experts
- Use emojis strategically (max 2 per tweet)
- Make it engaging, not academic

Format each tweet as:
TWEET 1: [content]
TWEET 2: [content]
etc.

Write the thread now:"""

    def _contrarian_prompt(self, topic: str, context: str, max_tweets: int) -> str:
        """Contrarian thread prompt."""
        return f"""Generate a contrarian Twitter thread about "{topic}" that challenges conventional wisdom.

{context}

Requirements:
- {max_tweets} tweets maximum
- Tweet 1: 🚨 Hot take that goes against popular opinion
- Tweets 2-{max_tweets-2}: Evidence from podcasts that supports your contrarian view
- Find conflicting viewpoints between different sources
- Highlight what mainstream investors/thinkers are missing
- Tweet {max_tweets-1}: Why this matters for investors/founders
- Tweet {max_tweets}: Challenge readers with a question
- Each tweet MUST be under 280 characters
- Be provocative but backed by data

Format each tweet as:
TWEET 1: [content]
TWEET 2: [content]
etc.

Write the thread now:"""

    def _hot_take_prompt(self, topic: str, context: str, max_tweets: int) -> str:
        """Hot take thread prompt."""
        return f"""Generate a spicy hot take Twitter thread about "{topic}".

{context}

Requirements:
- {max_tweets} tweets maximum
- Tweet 1: 🔥 Bold, controversial claim that stops scrolling
- Tweets 2-{max_tweets-2}: Support your hot take with podcast evidence
- Include "Everyone says X, but actually Y" moments
- Quote specific VCs/founders who disagree with each other
- Tweet {max_tweets-1}: Prediction or implication
- Tweet {max_tweets}: Ask readers to agree/disagree (drives engagement)
- Each tweet MUST be under 280 characters
- Be bold, not reckless

Format each tweet as:
TWEET 1: [content]
TWEET 2: [content]
etc.

Write the thread now:"""

    def _data_driven_prompt(self, topic: str, context: str, max_tweets: int) -> str:
        """Data-driven thread prompt."""
        return f"""Generate a data-driven Twitter thread about "{topic}" with stats and numbers.

{context}

Requirements:
- {max_tweets} tweets maximum
- Tweet 1: Lead with a surprising stat or number
- Tweets 2-{max_tweets-2}: More data points, valuations, metrics from podcasts
- Include specific numbers: valuations, ARR, growth rates, etc.
- Compare companies: "X raised $Y at Z valuation vs..."
- Tweet {max_tweets-1}: What the data means for investors
- Tweet {max_tweets}: Share your analysis or prediction
- Each tweet MUST be under 280 characters
- Cite sources when possible ("According to [Podcast/Guest]...")

Format each tweet as:
TWEET 1: [content]
TWEET 2: [content]
etc.

Write the thread now:"""

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate text using configured LLM with retry/fallback."""
        result = self.llm_client.chat(
            messages=[
                {"role": "system", "content": "You are a Twitter expert who creates viral threads for tech investors and founders."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return result if result else "Error: LLM returned no response"

    def _parse_thread(self, raw_thread: str) -> List[str]:
        """Parse thread into individual tweets."""
        # Try to find tweets in format "TWEET N: content"
        tweets = re.findall(r'TWEET \d+:\s*(.+?)(?=TWEET \d+:|$)', raw_thread, re.DOTALL)

        if not tweets:
            # Fallback: split by numbered format "1/", "2/", etc.
            tweets = re.split(r'\n\d+/', raw_thread)
            tweets = [t.strip() for t in tweets if t.strip()]

        if not tweets:
            # Last resort: split by newlines and filter
            tweets = [t.strip() for t in raw_thread.split('\n\n') if t.strip()]

        return tweets

    def _validate_tweets(self, tweets: List[str]) -> List[str]:
        """Validate and truncate tweets to 280 characters."""
        validated = []
        for tweet in tweets:
            # Remove any leading numbers/formatting
            tweet = re.sub(r'^\d+[/)\.]\s*', '', tweet)
            tweet = tweet.strip()

            # Truncate if too long
            if len(tweet) > 280:
                tweet = tweet[:277] + '...'

            if tweet:  # Only add non-empty tweets
                validated.append(tweet)

        return validated

    def _format_for_twitter(self, tweets: List[str]) -> str:
        """Format tweets for easy copy-paste to Twitter."""
        formatted = []
        for i, tweet in enumerate(tweets, 1):
            # Add thread numbering if not already present
            if not re.match(r'^\d+/', tweet):
                formatted.append(f"{i}/{len(tweets)} {tweet}")
            else:
                formatted.append(tweet)

        return '\n\n---\n\n'.join(formatted)

    def save_tweet(self, result: Dict[str, Any], output_dir: str = "outputs/tweets") -> str:
        """Save tweet or thread to file."""
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create filename
        date_str = datetime.now().strftime('%Y-%m-%d')
        topic_slug = re.sub(r'[^\w\s-]', '', result['topic'].lower())
        topic_slug = re.sub(r'[-\s]+', '-', topic_slug).strip('-')

        if result.get('mode') == 'single':
            # Single tweet
            filename = f"{date_str}-{topic_slug}-tweet.txt"
            file_path = output_path / filename
            content = f"""Tweet: {result['topic']}
Generated: {result['metadata']['generated_at']}
Sources: {result['source_count']} podcast episodes
Characters: {result['char_count']}

---

COPY-PASTE READY:

{result['tweet']}
"""
        else:
            # Thread
            style = result.get('style', 'educational')
            filename = f"{date_str}-{topic_slug}-{style}.txt"
            file_path = output_path / filename
            content = f"""Twitter Thread: {result['topic']}
Style: {style}
Generated: {result['metadata']['generated_at']}
Sources: {result['source_count']} podcast episodes
Tweets: {result['tweet_count']}

---

COPY-PASTE READY:

{result['copy_paste']}

---

INDIVIDUAL TWEETS:

"""
            for i, tweet_text in enumerate(result['tweets'], 1):
                content += f"\nTWEET {i}:\n{tweet_text}\n({len(tweet_text)} characters)\n"

        # Save
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return str(file_path)

