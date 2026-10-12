"""Storage management utilities for P³."""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from rich.console import Console


class StorageManager:
    """Manage storage and cleanup of audio files."""

    def __init__(self, audio_dir: str = "data/audio", db=None):
        """Initialize storage manager.

        Args:
            audio_dir: Directory containing audio files
            db: Database instance (optional, for status checks)
        """
        self.audio_dir = Path(audio_dir)
        self.db = db

    def cleanup_old_audio(self, retention_days: int = 7, dry_run: bool = False,
                         console: Console = None) -> Dict[str, any]:
        """Delete audio files older than retention period.

        Audio files are safe to delete after transcription since:
        - Full transcripts are stored in the database
        - Structured summaries are also stored
        - Original audio can be re-downloaded from RSS feed if needed

        Args:
            retention_days: Keep audio files for this many days after transcription.
                Use 0 to delete any audio that has a verified transcript.
            dry_run: If True, only report what would be deleted without deleting
            console: Rich console for output

        Returns:
            Dictionary with cleanup statistics
        """
        if not self.audio_dir.exists():
            return {
                "deleted_count": 0,
                "freed_space_mb": 0,
                "error": "Audio directory does not exist"
            }

        immediate_delete = retention_days <= 0
        if immediate_delete and not self.db:
            return {
                "deleted_count": 0,
                "freed_space_mb": 0,
                "error": "Database required to verify transcription for immediate deletion"
            }

        cutoff_date = None if immediate_delete else datetime.now() - timedelta(days=retention_days)

        deleted_count = 0
        freed_space_bytes = 0
        deleted_files = []

        # Get all audio files
        audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg']
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(self.audio_dir.glob(f"*{ext}"))

        # Check each file
        for file_path in audio_files:
            # Get file modification time
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

            # If file is older than retention period (or immediate deletion enabled)
            if immediate_delete or file_mtime < cutoff_date:
                # Check if episode is transcribed (if db available)
                if self.db:
                    episode_id = self._get_episode_id_from_filename(file_path.name)
                    if not episode_id or not self._is_transcribed(episode_id):
                        # Don't delete if not yet transcribed (or cannot verify)
                        continue

                file_size = file_path.stat().st_size
                freed_space_bytes += file_size

                if not dry_run:
                    try:
                        file_path.unlink()
                        deleted_files.append(file_path.name)
                        deleted_count += 1
                    except Exception as e:
                        if console:
                            console.print(f"[yellow]Warning: Could not delete {file_path.name}: {e}[/yellow]")
                else:
                    deleted_files.append(file_path.name)
                    deleted_count += 1

        freed_space_mb = freed_space_bytes / (1024 * 1024)

        return {
            "deleted_count": deleted_count,
            "freed_space_mb": round(freed_space_mb, 2),
            "deleted_files": deleted_files,
            "retention_days": retention_days,
            "dry_run": dry_run
        }

    def get_storage_stats(self) -> Dict[str, any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage information
        """
        if not self.audio_dir.exists():
            return {
                "total_files": 0,
                "total_size_mb": 0,
                "oldest_file_age_days": 0
            }

        audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg']
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(self.audio_dir.glob(f"*{ext}"))

        total_size_bytes = sum(f.stat().st_size for f in audio_files)
        total_size_mb = total_size_bytes / (1024 * 1024)

        # Find oldest file
        oldest_age_days = 0
        if audio_files:
            oldest_mtime = min(f.stat().st_mtime for f in audio_files)
            oldest_age_days = (datetime.now() - datetime.fromtimestamp(oldest_mtime)).days

        # Breakdown by status if db available
        transcribed_count = 0
        pending_count = 0

        if self.db:
            for file_path in audio_files:
                episode_id = self._get_episode_id_from_filename(file_path.name)
                if episode_id:
                    if self._is_transcribed(episode_id):
                        transcribed_count += 1
                    else:
                        pending_count += 1

        return {
            "total_files": len(audio_files),
            "total_size_mb": round(total_size_mb, 2),
            "oldest_file_age_days": oldest_age_days,
            "transcribed_count": transcribed_count,
            "pending_transcription": pending_count,
            "audio_directory": str(self.audio_dir)
        }

    def _get_episode_id_from_filename(self, filename: str) -> int:
        """Extract episode ID from audio filename.

        Audio files are typically named: {episode_id}_{title}.wav
        """
        try:
            # Try to extract ID from start of filename
            parts = filename.split('_')
            if parts and parts[0].isdigit():
                return int(parts[0])
        except Exception:
            pass
        return None

    def _is_transcribed(self, episode_id: int) -> bool:
        """Check if episode has been transcribed."""
        if not self.db:
            return False

        try:
            result = self.db.conn.execute(
                "SELECT COUNT(*) FROM transcripts WHERE episode_id = ?",
                [episode_id]
            ).fetchone()
            return result[0] > 0
        except Exception:
            return False

    def estimate_cleanup_impact(self, retention_days: int = 7) -> Dict[str, any]:
        """Estimate impact of cleanup without actually deleting.

        Args:
            retention_days: Retention period to simulate

        Returns:
            Dictionary with estimated impact
        """
        return self.cleanup_old_audio(retention_days=retention_days, dry_run=True)

