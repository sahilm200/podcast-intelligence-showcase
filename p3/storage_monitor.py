"""Real-time storage monitoring for audio digest generation."""

from __future__ import annotations

import shutil
import time
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class StorageMonitor:
    """Monitor disk storage during audio digest generation."""

    def __init__(self, path: str = ".", min_free_gb: float = 1.0, warn_threshold_gb: float = 2.0):
        """Initialize storage monitor.

        Args:
            path: Path to monitor (defaults to current directory)
            min_free_gb: Minimum free space required in GB
            warn_threshold_gb: Warn when free space drops below this GB
        """
        self.path = path
        self.min_free_gb = min_free_gb
        self.warn_threshold_gb = warn_threshold_gb
        self.initial_free_gb = 0
        self.last_check_time = 0
        self.check_interval = 5  # Check every 5 seconds

    def get_disk_usage(self) -> Dict[str, float]:
        """Get current disk usage statistics.

        Returns:
            Dictionary with disk usage stats in GB
        """
        try:
            stat = shutil.disk_usage(self.path)
            return {
                'total_gb': stat.total / (1024 ** 3),
                'used_gb': stat.used / (1024 ** 3),
                'free_gb': stat.free / (1024 ** 3),
                'used_percent': (stat.used / stat.total) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
            return {
                'total_gb': 0,
                'used_gb': 0,
                'free_gb': 0,
                'used_percent': 0
            }

    def check_available_space(self) -> tuple[bool, Optional[str]]:
        """Check if sufficient disk space is available.

        Returns:
            Tuple of (is_sufficient, warning_message)
        """
        usage = self.get_disk_usage()
        free_gb = usage['free_gb']

        if free_gb < self.min_free_gb:
            return False, f"Insufficient disk space: {free_gb:.1f} GB free (need {self.min_free_gb} GB minimum)"

        if free_gb < self.warn_threshold_gb:
            return True, f"Low disk space warning: {free_gb:.1f} GB free (recommend {self.warn_threshold_gb}+ GB)"

        return True, None

    def start_monitoring(self):
        """Start monitoring session by recording initial state."""
        usage = self.get_disk_usage()
        self.initial_free_gb = usage['free_gb']
        self.last_check_time = time.time()
        logger.info(f"Storage monitoring started: {self.initial_free_gb:.1f} GB free")

    def check_during_generation(self, force: bool = False) -> tuple[bool, Optional[str]]:
        """Check disk space during generation (throttled).

        Only checks if enough time has passed since last check,
        unless force=True.

        Args:
            force: Force check regardless of time interval

        Returns:
            Tuple of (is_ok, warning_message)
        """
        current_time = time.time()
        if not force and (current_time - self.last_check_time) < self.check_interval:
            # Too soon since last check
            return True, None

        self.last_check_time = current_time
        return self.check_available_space()

    def get_space_used_since_start(self) -> float:
        """Get amount of space used since monitoring started.

        Returns:
            Space used in GB (can be negative if space was freed)
        """
        if self.initial_free_gb == 0:
            return 0

        current_usage = self.get_disk_usage()
        current_free = current_usage['free_gb']

        # Positive value means we used space, negative means we freed space
        return self.initial_free_gb - current_free

    def get_monitoring_summary(self) -> Dict[str, any]:
        """Get summary of storage monitoring session.

        Returns:
            Dictionary with monitoring summary
        """
        current_usage = self.get_disk_usage()
        space_used = self.get_space_used_since_start()

        return {
            'initial_free_gb': round(self.initial_free_gb, 2),
            'current_free_gb': round(current_usage['free_gb'], 2),
            'space_used_gb': round(space_used, 2),
            'total_gb': round(current_usage['total_gb'], 2),
            'used_percent': round(current_usage['used_percent'], 1)
        }

    def format_summary(self) -> str:
        """Format monitoring summary as human-readable string.

        Returns:
            Formatted summary string
        """
        summary = self.get_monitoring_summary()

        if summary['space_used_gb'] > 0:
            return (f"Storage: {summary['current_free_gb']} GB free "
                   f"({summary['space_used_gb']} GB used during generation)")
        else:
            return f"Storage: {summary['current_free_gb']} GB free"


class StorageWarning(Exception):
    """Raised when storage space is critically low."""
    pass


def check_storage_before_operation(required_gb: float = 1.0, path: str = ".") -> bool:
    """Quick check if enough storage is available before starting operation.

    Args:
        required_gb: Required free space in GB
        path: Path to check

    Returns:
        True if sufficient space available

    Raises:
        StorageWarning: If insufficient space available
    """
    monitor = StorageMonitor(path=path, min_free_gb=required_gb)
    is_sufficient, message = monitor.check_available_space()

    if not is_sufficient:
        raise StorageWarning(message)

    if message:
        # Warning but not critical
        logger.warning(message)

    return True

