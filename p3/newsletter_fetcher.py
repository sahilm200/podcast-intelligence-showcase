"""Substack newsletter fetcher for P³."""

import feedparser
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from .database import P3Database


class SubstackFetcher:
    def __init__(self, db: P3Database):
        self.db = db

    def _normalize_substack_url(self, url: str) -> str:
        """Normalize Substack URL to base domain."""
        parsed = urlparse(url)
        # Extract base domain (e.g., https://stratechery.substack.com)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        return base_url

    def _get_rss_url(self, substack_url: str) -> str:
        """Get RSS feed URL from Substack URL."""
        base_url = self._normalize_substack_url(substack_url)
        return f"{base_url}/feed"

    def _extract_newsletter_info(self, feed_data: Any) -> Dict[str, str]:
        """Extract newsletter metadata from RSS feed."""
        feed_info = feed_data.feed
        return {
            "name": feed_info.get("title", "Unknown Newsletter"),
            "author": feed_info.get("author", "Unknown Author")
        }

    def _extract_article_content(self, url: str, rss_content: str = None) -> str:
        """Extract article content from URL or RSS."""
        # First try using RSS content if available
        if rss_content:
            soup = BeautifulSoup(rss_content, 'html.parser')
            # Remove any script/style tags
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n', strip=True)
            if len(text) > 200:  # RSS content is substantial
                return text

        # Otherwise, fetch from web
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Substack articles are typically in div with class 'available-content'
                    # or within article tags
                    article_content = soup.find('div', class_='available-content')
                    if not article_content:
                        article_content = soup.find('article')
                    if not article_content:
                        article_content = soup.find('div', class_='body')

                    if article_content:
                        # Remove script/style tags
                        for script in article_content(["script", "style"]):
                            script.decompose()
                        return article_content.get_text(separator='\n', strip=True)
        except Exception as e:
            print(f"Failed to fetch article content from web: {e}")

        return rss_content or ""

    def add_newsletter(self, substack_url: str) -> int:
        """Add a new newsletter to track."""
        base_url = self._normalize_substack_url(substack_url)
        rss_url = self._get_rss_url(base_url)

        # Check if newsletter already exists
        existing = self.db.get_newsletter_by_url(base_url)
        if existing:
            print(f"Newsletter already exists: {existing['name']}")
            return existing['id']

        # Fetch RSS to get metadata
        try:
            feed_data = feedparser.parse(rss_url)
            if feed_data.bozo and not feed_data.entries:
                raise Exception(f"Invalid RSS feed: {rss_url}")

            info = self._extract_newsletter_info(feed_data)
            newsletter_id = self.db.add_newsletter(
                name=info['name'],
                substack_url=base_url,
                rss_url=rss_url,
                author=info['author']
            )
            print(f"✓ Added newsletter: {info['name']} by {info['author']}")
            return newsletter_id

        except Exception as e:
            print(f"✗ Failed to add newsletter: {e}")
            raise

    def fetch_articles(self, newsletter_id: int = None, max_articles: int = 10) -> int:
        """Fetch recent articles from newsletters."""
        newsletters = []
        if newsletter_id:
            # Fetch specific newsletter
            result = self.db.conn.execute(
                "SELECT * FROM newsletters WHERE id = ?", (newsletter_id,)
            ).fetchone()
            if result:
                newsletters.append({
                    "id": result[0],
                    "name": result[1],
                    "substack_url": result[2],
                    "rss_url": result[3]
                })
        else:
            # Fetch all newsletters
            newsletters = self.db.get_all_newsletters()

        total_fetched = 0
        for newsletter in newsletters:
            print(f"\nFetching articles from: {newsletter['name']}")
            try:
                feed_data = feedparser.parse(newsletter['rss_url'])
                count = 0

                for entry in feed_data.entries[:max_articles]:
                    article_url = entry.get('link', '')

                    # Skip if already exists
                    if self.db.article_exists(article_url):
                        continue

                    title = entry.get('title', 'Untitled')
                    published = entry.get('published_parsed')

                    # Parse published date with error handling
                    try:
                        published_date = datetime(*published[:6]) if published else datetime.now()
                    except (TypeError, ValueError):
                        # Fallback to current time if parsing fails
                        published_date = datetime.now()

                    # Extract content
                    rss_content = entry.get('content', [{}])[0].get('value', '') or entry.get('description', '')
                    content = self._extract_article_content(article_url, rss_content)

                    # Store article
                    self.db.add_newsletter_article(
                        newsletter_id=newsletter['id'],
                        title=title,
                        published_date=published_date,
                        url=article_url,
                        content=content
                    )
                    count += 1
                    print(f"  ✓ {title}")

                print(f"Fetched {count} new articles from {newsletter['name']}")
                total_fetched += count

            except Exception as e:
                print(f"✗ Failed to fetch from {newsletter['name']}: {e}")

        return total_fetched

    def list_newsletters(self) -> List[Dict[str, Any]]:
        """List all tracked newsletters."""
        return self.db.get_all_newsletters()

