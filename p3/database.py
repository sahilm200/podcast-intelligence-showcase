"""Database layer using DuckDB for P³ storage."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import duckdb
from pathlib import Path


class P3Database:
    def __init__(self, db_path: str = "data/p3.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_schema()

    def _initialize_schema(self):
        """Create database schema if not exists."""
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS podcast_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS podcasts (
                id INTEGER PRIMARY KEY DEFAULT nextval('podcast_id_seq'),
                title VARCHAR NOT NULL,
                rss_url VARCHAR UNIQUE NOT NULL,
                category VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS episode_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY DEFAULT nextval('episode_id_seq'),
                podcast_id INTEGER REFERENCES podcasts(id),
                title VARCHAR NOT NULL,
                date TIMESTAMP,
                url VARCHAR UNIQUE NOT NULL,
                file_path VARCHAR,
                duration_seconds INTEGER,
                status VARCHAR DEFAULT 'downloaded', -- downloaded, transcribed, processed
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS transcript_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY DEFAULT nextval('transcript_id_seq'),
                episode_id INTEGER REFERENCES episodes(id),
                speaker VARCHAR,
                timestamp_start REAL,
                timestamp_end REAL,
                text TEXT NOT NULL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS summary_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY DEFAULT nextval('summary_id_seq'),
                episode_id INTEGER REFERENCES episodes(id),
                key_topics JSON,
                themes JSON,
                quotes JSON,
                startups JSON,
                digest_date DATE,
                full_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: Add new fields if they don't exist
        try:
            self.conn.execute("""
                ALTER TABLE summaries ADD COLUMN IF NOT EXISTS key_insights JSON
            """)
            self.conn.execute("""
                ALTER TABLE summaries ADD COLUMN IF NOT EXISTS actionable_takeaways JSON
            """)
        except Exception:
            # Columns might already exist, that's fine
            pass

        # Newsletter tables
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS newsletter_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS newsletters (
                id INTEGER PRIMARY KEY DEFAULT nextval('newsletter_id_seq'),
                name VARCHAR NOT NULL,
                substack_url VARCHAR UNIQUE NOT NULL,
                rss_url VARCHAR,
                author VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS article_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS newsletter_articles (
                id INTEGER PRIMARY KEY DEFAULT nextval('article_id_seq'),
                newsletter_id INTEGER REFERENCES newsletters(id),
                title VARCHAR NOT NULL,
                published_date TIMESTAMP,
                url VARCHAR UNIQUE NOT NULL,
                content TEXT,
                status VARCHAR DEFAULT 'fetched', -- fetched, processed
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS newsletter_summary_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS newsletter_summaries (
                id INTEGER PRIMARY KEY DEFAULT nextval('newsletter_summary_id_seq'),
                article_id INTEGER REFERENCES newsletter_articles(id),
                key_topics JSON,
                themes JSON,
                quotes JSON,
                companies JSON,
                digest_date DATE,
                full_summary TEXT,
                key_insights JSON,
                actionable_takeaways JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Audio digest tables
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS audio_clip_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_clips (
                id INTEGER PRIMARY KEY DEFAULT nextval('audio_clip_id_seq'),
                episode_id INTEGER REFERENCES episodes(id),
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                duration_sec REAL NOT NULL,
                text TEXT,
                topic VARCHAR,
                score INTEGER,
                file_path VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS audio_digest_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_digests (
                id INTEGER PRIMARY KEY DEFAULT nextval('audio_digest_id_seq'),
                digest_date DATE NOT NULL,
                file_path VARCHAR NOT NULL,
                duration_sec REAL,
                target_duration_sec INTEGER,
                clip_count INTEGER,
                episode_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Sync sequences to avoid duplicate key errors after external modifications
        self._sync_sequences()

    def _sync_sequences(self):
        """Reset sequences to MAX(id)+1 to prevent duplicate key errors."""
        seq_table_pairs = [
            ("podcast_id_seq", "podcasts"),
            ("episode_id_seq", "episodes"),
            ("transcript_id_seq", "transcripts"),
            ("summary_id_seq", "summaries"),
            ("newsletter_id_seq", "newsletters"),
            ("article_id_seq", "newsletter_articles"),
            ("newsletter_summary_id_seq", "newsletter_summaries"),
            ("audio_clip_id_seq", "audio_clips"),
            ("audio_digest_id_seq", "audio_digests"),
        ]
        for seq_name, table_name in seq_table_pairs:
            try:
                max_id = self.conn.execute(
                    f"SELECT COALESCE(MAX(id), 0) FROM {table_name}"
                ).fetchone()[0]
                if max_id > 0:
                    # Drop and recreate the sequence starting after the max id
                    self.conn.execute(f"DROP SEQUENCE IF EXISTS {seq_name}")
                    self.conn.execute(
                        f"CREATE SEQUENCE {seq_name} START {max_id + 1}"
                    )
            except Exception:
                pass

    def add_podcast(self, title: str, rss_url: str, category: str = None) -> int:
        """Add new podcast feed."""
        # Get the next ID first
        next_id = self.conn.execute("SELECT nextval('podcast_id_seq')").fetchone()[0]
        self.conn.execute(
            "INSERT INTO podcasts (id, title, rss_url, category) VALUES (?, ?, ?, ?)",
            (next_id, title, rss_url, category)
        )
        return next_id

    def get_podcast_by_url(self, rss_url: str) -> Optional[Dict[str, Any]]:
        """Get podcast by RSS URL."""
        result = self.conn.execute(
            "SELECT * FROM podcasts WHERE rss_url = ?", (rss_url,)
        ).fetchone()
        if result:
            return {
                "id": result[0],
                "title": result[1],
                "rss_url": result[2],
                "category": result[3],
                "created_at": result[4]
            }
        return None

    def add_episode(self, podcast_id: int, title: str, date: datetime, url: str, 
                   file_path: str = None) -> int:
        """Add new episode."""
        next_id = self.conn.execute("SELECT nextval('episode_id_seq')").fetchone()[0]
        self.conn.execute("""
            INSERT INTO episodes (id, podcast_id, title, date, url, file_path) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (next_id, podcast_id, title, date, url, file_path))
        return next_id

    def episode_exists(self, url: str) -> bool:
        """Check if episode already exists."""
        result = self.conn.execute(
            "SELECT 1 FROM episodes WHERE url = ?", (url,)
        ).fetchone()
        return result is not None

    def get_episodes_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get episodes by processing status."""
        results = self.conn.execute("""
            SELECT e.*, p.title as podcast_title 
            FROM episodes e 
            JOIN podcasts p ON e.podcast_id = p.id 
            WHERE e.status = ?
            ORDER BY e.date DESC
        """, (status,)).fetchall()
        
        episodes = []
        for row in results:
            episodes.append({
                "id": row[0],
                "podcast_id": row[1],
                "title": row[2],
                "date": row[3],
                "url": row[4],
                "file_path": row[5],
                "duration_seconds": row[6],
                "status": row[7],
                "created_at": row[8],
                "podcast_title": row[9]
            })
        return episodes

    def update_episode_status(self, episode_id: int, status: str):
        """Update episode processing status."""
        self.conn.execute(
            "UPDATE episodes SET status = ? WHERE id = ?",
            (status, episode_id)
        )

    def add_transcript_segments(self, episode_id: int, segments: List[Dict[str, Any]]):
        """Add transcript segments for an episode."""
        for segment in segments:
            self.conn.execute("""
                INSERT INTO transcripts (episode_id, speaker, timestamp_start, timestamp_end, text, confidence) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                episode_id,
                segment.get("speaker"),
                segment.get("start"),
                segment.get("end"),
                segment.get("text"),
                segment.get("confidence")
            ))

    def get_transcripts_for_episode(self, episode_id: int) -> List[Dict[str, Any]]:
        """Get all transcript segments for an episode."""
        results = self.conn.execute("""
            SELECT * FROM transcripts WHERE episode_id = ? 
            ORDER BY timestamp_start
        """, (episode_id,)).fetchall()
        
        transcripts = []
        for row in results:
            transcripts.append({
                "id": row[0],
                "episode_id": row[1],
                "speaker": row[2],
                "timestamp_start": row[3],
                "timestamp_end": row[4],
                "text": row[5],
                "confidence": row[6],
                "created_at": row[7]
            })
        return transcripts

    def add_summary(self, episode_id: int, key_topics: List[str], themes: List[str],
                   quotes: List[str], startups: List[str], full_summary: str,
                   digest_date: datetime = None, key_insights: List[str] = None,
                   actionable_takeaways: List[str] = None):
        """Add episode summary."""
        if digest_date is None:
            digest_date = datetime.now().date()
        if key_insights is None:
            key_insights = []
        if actionable_takeaways is None:
            actionable_takeaways = []

        import json
        self.conn.execute("""
            INSERT INTO summaries (episode_id, key_topics, themes, quotes, startups,
                                  full_summary, digest_date, key_insights, actionable_takeaways)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode_id,
            json.dumps(key_topics),
            json.dumps(themes),
            json.dumps(quotes),
            json.dumps(startups),
            full_summary,
            digest_date,
            json.dumps(key_insights),
            json.dumps(actionable_takeaways)
        ))

    def get_summaries_by_date(self, date: datetime) -> List[Dict[str, Any]]:
        """Get all summaries for a specific date."""
        results = self.conn.execute("""
            SELECT s.id, s.episode_id, s.key_topics, s.themes, s.quotes, s.startups,
                   s.digest_date, s.full_summary, s.created_at, s.key_insights, s.actionable_takeaways,
                   e.title as episode_title, p.title as podcast_title
            FROM summaries s
            JOIN episodes e ON s.episode_id = e.id
            JOIN podcasts p ON e.podcast_id = p.id
            WHERE s.digest_date = ?
            ORDER BY p.title, e.title
        """, (date.date(),)).fetchall()

        summaries = []
        for row in results:
            import json
            summaries.append({
                "id": row[0],
                "episode_id": row[1],
                "key_topics": json.loads(row[2]) if row[2] else [],
                "themes": json.loads(row[3]) if row[3] else [],
                "quotes": json.loads(row[4]) if row[4] else [],
                "startups": json.loads(row[5]) if row[5] else [],
                "digest_date": row[6],
                "full_summary": row[7],
                "created_at": row[8],
                "key_insights": json.loads(row[9]) if row[9] else [],
                "actionable_takeaways": json.loads(row[10]) if row[10] else [],
                "episode_title": row[11],
                "podcast_title": row[12]
            })
        return summaries

    def get_recent_summaries(self, lookback_days: int = 7) -> List[Dict[str, Any]]:
        """Get all summaries from the last N days.

        Args:
            lookback_days: Number of days to look back (default: 7)

        Returns:
            List of summary dicts ordered by digest date descending
        """
        lookback_days = int(lookback_days)
        results = self.conn.execute("""
            SELECT s.id, s.episode_id, s.key_topics, s.themes, s.quotes, s.startups,
                   s.digest_date, s.full_summary, s.created_at, s.key_insights, s.actionable_takeaways,
                   e.title as episode_title, p.title as podcast_title
            FROM summaries s
            JOIN episodes e ON s.episode_id = e.id
            JOIN podcasts p ON e.podcast_id = p.id
            WHERE s.digest_date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
            ORDER BY s.digest_date DESC, p.title, e.title
        """, (lookback_days,)).fetchall()

        summaries = []
        for row in results:
            import json
            summaries.append({
                "id": row[0],
                "episode_id": row[1],
                "key_topics": json.loads(row[2]) if row[2] else [],
                "themes": json.loads(row[3]) if row[3] else [],
                "quotes": json.loads(row[4]) if row[4] else [],
                "startups": json.loads(row[5]) if row[5] else [],
                "digest_date": row[6],
                "full_summary": row[7],
                "created_at": row[8],
                "key_insights": json.loads(row[9]) if row[9] else [],
                "actionable_takeaways": json.loads(row[10]) if row[10] else [],
                "episode_title": row[11],
                "podcast_title": row[12]
            })
        return summaries

    # Newsletter methods
    def add_newsletter(self, name: str, substack_url: str, rss_url: str = None, author: str = None) -> int:
        """Add new newsletter."""
        next_id = self.conn.execute("SELECT nextval('newsletter_id_seq')").fetchone()[0]
        self.conn.execute(
            "INSERT INTO newsletters (id, name, substack_url, rss_url, author) VALUES (?, ?, ?, ?, ?)",
            (next_id, name, substack_url, rss_url, author)
        )
        return next_id

    def get_newsletter_by_url(self, substack_url: str) -> Optional[Dict[str, Any]]:
        """Get newsletter by Substack URL."""
        result = self.conn.execute(
            "SELECT * FROM newsletters WHERE substack_url = ?", (substack_url,)
        ).fetchone()
        if result:
            return {
                "id": result[0],
                "name": result[1],
                "substack_url": result[2],
                "rss_url": result[3],
                "author": result[4],
                "created_at": result[5]
            }
        return None

    def get_all_newsletters(self) -> List[Dict[str, Any]]:
        """Get all newsletters."""
        results = self.conn.execute("SELECT * FROM newsletters ORDER BY name").fetchall()
        newsletters = []
        for row in results:
            newsletters.append({
                "id": row[0],
                "name": row[1],
                "substack_url": row[2],
                "rss_url": row[3],
                "author": row[4],
                "created_at": row[5]
            })
        return newsletters

    def add_newsletter_article(self, newsletter_id: int, title: str, published_date: datetime,
                               url: str, content: str = None) -> int:
        """Add new newsletter article."""
        next_id = self.conn.execute("SELECT nextval('article_id_seq')").fetchone()[0]
        self.conn.execute("""
            INSERT INTO newsletter_articles (id, newsletter_id, title, published_date, url, content)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (next_id, newsletter_id, title, published_date, url, content))
        return next_id

    def article_exists(self, url: str) -> bool:
        """Check if newsletter article already exists."""
        result = self.conn.execute(
            "SELECT 1 FROM newsletter_articles WHERE url = ?", (url,)
        ).fetchone()
        return result is not None

    def get_articles_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get newsletter articles by processing status."""
        results = self.conn.execute("""
            SELECT a.*, n.name as newsletter_name
            FROM newsletter_articles a
            JOIN newsletters n ON a.newsletter_id = n.id
            WHERE a.status = ?
            ORDER BY a.published_date DESC
        """, (status,)).fetchall()

        articles = []
        for row in results:
            articles.append({
                "id": row[0],
                "newsletter_id": row[1],
                "title": row[2],
                "published_date": row[3],
                "url": row[4],
                "content": row[5],
                "status": row[6],
                "created_at": row[7],
                "newsletter_name": row[8]
            })
        return articles

    def update_article_status(self, article_id: int, status: str):
        """Update newsletter article processing status."""
        self.conn.execute(
            "UPDATE newsletter_articles SET status = ? WHERE id = ?",
            (status, article_id)
        )

    def add_newsletter_summary(self, article_id: int, key_topics: List[str], themes: List[str],
                               quotes: List[str], companies: List[str], full_summary: str,
                               digest_date: datetime = None, key_insights: List[str] = None,
                               actionable_takeaways: List[str] = None):
        """Add newsletter article summary."""
        if digest_date is None:
            digest_date = datetime.now().date()
        if key_insights is None:
            key_insights = []
        if actionable_takeaways is None:
            actionable_takeaways = []

        import json
        self.conn.execute("""
            INSERT INTO newsletter_summaries (article_id, key_topics, themes, quotes, companies,
                                             full_summary, digest_date, key_insights, actionable_takeaways)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article_id,
            json.dumps(key_topics),
            json.dumps(themes),
            json.dumps(quotes),
            json.dumps(companies),
            full_summary,
            digest_date,
            json.dumps(key_insights),
            json.dumps(actionable_takeaways)
        ))

    def get_newsletter_summaries_by_date(self, date: datetime) -> List[Dict[str, Any]]:
        """Get all newsletter summaries for a specific date."""
        # Handle both datetime and date objects
        from datetime import date as date_type
        if isinstance(date, date_type) and not isinstance(date, datetime):
            query_date = date
        else:
            query_date = date.date()

        results = self.conn.execute("""
            SELECT s.id, s.article_id, s.key_topics, s.themes, s.quotes, s.companies,
                   s.digest_date, s.full_summary, s.created_at, s.key_insights, s.actionable_takeaways,
                   a.title as article_title, n.name as newsletter_name, a.url as article_url
            FROM newsletter_summaries s
            JOIN newsletter_articles a ON s.article_id = a.id
            JOIN newsletters n ON a.newsletter_id = n.id
            WHERE s.digest_date = ?
            ORDER BY n.name, a.title
        """, (query_date,)).fetchall()

        summaries = []
        for row in results:
            import json
            summaries.append({
                "id": row[0],
                "article_id": row[1],
                "key_topics": json.loads(row[2]) if row[2] else [],
                "themes": json.loads(row[3]) if row[3] else [],
                "quotes": json.loads(row[4]) if row[4] else [],
                "companies": json.loads(row[5]) if row[5] else [],
                "digest_date": row[6],
                "full_summary": row[7],
                "created_at": row[8],
                "key_insights": json.loads(row[9]) if row[9] else [],
                "actionable_takeaways": json.loads(row[10]) if row[10] else [],
                "article_title": row[11],
                "newsletter_name": row[12],
                "article_url": row[13]
            })
        return summaries

    def get_recent_newsletter_summaries(self, lookback_days: int = 7) -> List[Dict[str, Any]]:
        """Get all newsletter summaries from the last N days.

        Args:
            lookback_days: Number of days to look back (default: 7)

        Returns:
            List of summary dicts with fields compatible with podcast summaries
        """
        lookback_days = int(lookback_days)
        results = self.conn.execute("""
            SELECT s.id, s.article_id, s.key_topics, s.themes, s.quotes, s.companies,
                   s.digest_date, s.full_summary, s.created_at, s.key_insights, s.actionable_takeaways,
                   a.title as article_title, n.name as newsletter_name, a.url as article_url
            FROM newsletter_summaries s
            JOIN newsletter_articles a ON s.article_id = a.id
            JOIN newsletters n ON a.newsletter_id = n.id
            WHERE s.digest_date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
            ORDER BY s.digest_date DESC, n.name, a.title
        """, (lookback_days,)).fetchall()

        summaries = []
        for row in results:
            import json
            summaries.append({
                "id": row[0],
                "article_id": row[1],
                "key_topics": json.loads(row[2]) if row[2] else [],
                "themes": json.loads(row[3]) if row[3] else [],
                "quotes": json.loads(row[4]) if row[4] else [],
                "startups": json.loads(row[5]) if row[5] else [],
                "digest_date": row[6],
                "full_summary": row[7],
                "created_at": row[8],
                "key_insights": json.loads(row[9]) if row[9] else [],
                "actionable_takeaways": json.loads(row[10]) if row[10] else [],
                "episode_title": row[11],
                "podcast_title": f"[Newsletter] {row[12]}",
                "article_url": row[13],
                "source_type": "newsletter"
            })
        return summaries

    # Audio digest methods
    def add_audio_clip(self, episode_id: int, start_sec: float, end_sec: float,
                       text: str = None, topic: str = None, score: int = None,
                       file_path: str = None) -> int:
        """Add an audio clip."""
        duration_sec = end_sec - start_sec
        next_id = self.conn.execute("SELECT nextval('audio_clip_id_seq')").fetchone()[0]
        self.conn.execute("""
            INSERT INTO audio_clips (id, episode_id, start_sec, end_sec, duration_sec, text, topic, score, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (next_id, episode_id, start_sec, end_sec, duration_sec, text, topic, score, file_path))
        return next_id

    def get_clips_for_episode(self, episode_id: int) -> List[Dict[str, Any]]:
        """Get all audio clips for an episode."""
        results = self.conn.execute("""
            SELECT * FROM audio_clips WHERE episode_id = ?
            ORDER BY start_sec
        """, (episode_id,)).fetchall()

        clips = []
        for row in results:
            clips.append({
                "id": row[0],
                "episode_id": row[1],
                "start_sec": row[2],
                "end_sec": row[3],
                "duration_sec": row[4],
                "text": row[5],
                "topic": row[6],
                "score": row[7],
                "file_path": row[8],
                "created_at": row[9]
            })
        return clips

    def get_clips_by_date(self, date: datetime) -> List[Dict[str, Any]]:
        """Get all audio clips for episodes from a specific date."""
        results = self.conn.execute("""
            SELECT c.*, e.title as episode_title, p.title as podcast_title, e.file_path as episode_audio_path
            FROM audio_clips c
            JOIN episodes e ON c.episode_id = e.id
            JOIN podcasts p ON e.podcast_id = p.id
            WHERE DATE(e.date) = ?
            ORDER BY e.id, c.start_sec
        """, (date.date(),)).fetchall()

        clips = []
        for row in results:
            clips.append({
                "id": row[0],
                "episode_id": row[1],
                "start_sec": row[2],
                "end_sec": row[3],
                "duration_sec": row[4],
                "text": row[5],
                "topic": row[6],
                "score": row[7],
                "file_path": row[8],
                "created_at": row[9],
                "episode_title": row[10],
                "podcast_title": row[11],
                "episode_audio_path": row[12]
            })
        return clips

    def add_audio_digest(self, digest_date: datetime, file_path: str, duration_sec: float = None,
                        target_duration_sec: int = None, clip_count: int = None,
                        episode_count: int = None) -> int:
        """Add an audio digest."""
        next_id = self.conn.execute("SELECT nextval('audio_digest_id_seq')").fetchone()[0]
        self.conn.execute("""
            INSERT INTO audio_digests (id, digest_date, file_path, duration_sec, target_duration_sec,
                                      clip_count, episode_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (next_id, digest_date.date(), file_path, duration_sec, target_duration_sec, clip_count, episode_count))
        return next_id

    def get_audio_digest_by_date(self, date: datetime) -> Optional[Dict[str, Any]]:
        """Get audio digest for a specific date."""
        result = self.conn.execute("""
            SELECT * FROM audio_digests WHERE digest_date = ?
            ORDER BY created_at DESC LIMIT 1
        """, (date.date(),)).fetchone()

        if result:
            return {
                "id": result[0],
                "digest_date": result[1],
                "file_path": result[2],
                "duration_sec": result[3],
                "target_duration_sec": result[4],
                "clip_count": result[5],
                "episode_count": result[6],
                "created_at": result[7]
            }
        return None

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

