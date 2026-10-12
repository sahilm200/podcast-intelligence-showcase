"""Real-time performance monitoring for P3 operations."""

from __future__ import annotations

import time
import logging
from typing import Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


def _import_psutil():
    """Lazy import psutil to avoid breaking CLI if not installed.

    Returns:
        psutil module or None if not available
    """
    try:
        import psutil
        return psutil
    except ImportError:
        logger.warning("psutil not installed - performance monitoring disabled")
        return None


class PerformanceMonitor:
    """Monitor CPU, RAM, and disk I/O during operations."""

    def __init__(self, sample_interval: float = 5.0, history_size: int = 20):
        """Initialize performance monitor.

        Args:
            sample_interval: Seconds between samples
            history_size: Number of samples to keep in history
        """
        self.sample_interval = sample_interval
        self.history_size = history_size

        self.cpu_history = deque(maxlen=history_size)
        self.ram_history = deque(maxlen=history_size)
        self.disk_history = deque(maxlen=history_size)

        self.start_time = None
        self.last_sample_time = 0

        self.initial_stats = None

    def start(self):
        """Start monitoring session."""
        psutil = _import_psutil()
        if not psutil:
            logger.warning("Performance monitoring unavailable - psutil not installed")
            return

        self.start_time = time.time()
        self.last_sample_time = 0

        # Record initial stats
        self.initial_stats = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_available_gb': psutil.virtual_memory().available / (1024 ** 3),
            'disk_read_mb': psutil.disk_io_counters().read_bytes / (1024 ** 2) if psutil.disk_io_counters() else 0,
            'disk_write_mb': psutil.disk_io_counters().write_bytes / (1024 ** 2) if psutil.disk_io_counters() else 0,
        }

        logger.info(f"Performance monitoring started")

    def sample(self, force: bool = False) -> Optional[Dict[str, float]]:
        """Take a performance sample (throttled).

        Args:
            force: Force sample regardless of time interval

        Returns:
            Dictionary with current performance metrics, or None if too soon
        """
        psutil = _import_psutil()
        if not psutil:
            return None

        current_time = time.time()

        if not force and (current_time - self.last_sample_time) < self.sample_interval:
            return None

        self.last_sample_time = current_time

        # Get current stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()

        sample = {
            'timestamp': current_time,
            'cpu_percent': cpu_percent,
            'ram_percent': mem.percent,
            'ram_available_gb': mem.available / (1024 ** 3),
            'ram_used_gb': mem.used / (1024 ** 3),
            'disk_read_mb': disk_io.read_bytes / (1024 ** 2) if disk_io else 0,
            'disk_write_mb': disk_io.write_bytes / (1024 ** 2) if disk_io else 0,
        }

        # Add to history
        self.cpu_history.append(cpu_percent)
        self.ram_history.append(mem.percent)

        if disk_io:
            disk_total_mb = (disk_io.read_bytes + disk_io.write_bytes) / (1024 ** 2)
            self.disk_history.append(disk_total_mb)

        return sample

    def get_current_stats(self) -> Dict[str, any]:
        """Get current performance statistics.

        Returns:
            Dictionary with current performance metrics
        """
        # Force a sample
        sample = self.sample(force=True)

        if not sample:
            return {}

        # Calculate averages
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        avg_ram = sum(self.ram_history) / len(self.ram_history) if self.ram_history else 0
        max_cpu = max(self.cpu_history) if self.cpu_history else 0
        max_ram = max(self.ram_history) if self.ram_history else 0

        return {
            'current': sample,
            'averages': {
                'cpu_percent': round(avg_cpu, 1),
                'ram_percent': round(avg_ram, 1),
            },
            'peaks': {
                'cpu_percent': round(max_cpu, 1),
                'ram_percent': round(max_ram, 1),
            },
            'elapsed_sec': time.time() - self.start_time if self.start_time else 0
        }

    def get_summary(self) -> Dict[str, any]:
        """Get performance monitoring summary.

        Returns:
            Dictionary with performance summary
        """
        if not self.start_time:
            return {}

        stats = self.get_current_stats()
        current = stats.get('current', {})

        # Calculate disk I/O delta
        disk_read_delta = 0
        disk_write_delta = 0

        if self.initial_stats:
            disk_read_delta = current.get('disk_read_mb', 0) - self.initial_stats.get('disk_read_mb', 0)
            disk_write_delta = current.get('disk_write_mb', 0) - self.initial_stats.get('disk_write_mb', 0)

        return {
            'duration_sec': time.time() - self.start_time,
            'avg_cpu_percent': stats.get('averages', {}).get('cpu_percent', 0),
            'peak_cpu_percent': stats.get('peaks', {}).get('cpu_percent', 0),
            'avg_ram_percent': stats.get('averages', {}).get('ram_percent', 0),
            'peak_ram_percent': stats.get('peaks', {}).get('ram_percent', 0),
            'current_ram_available_gb': current.get('ram_available_gb', 0),
            'disk_read_mb': disk_read_delta,
            'disk_write_mb': disk_write_delta,
            'sample_count': len(self.cpu_history)
        }

    def format_summary(self) -> str:
        """Format performance summary as human-readable string.

        Returns:
            Formatted summary string
        """
        summary = self.get_summary()

        if not summary:
            return "Performance monitoring not started"

        lines = [
            f"Performance: CPU avg {summary['avg_cpu_percent']:.0f}% (peak {summary['peak_cpu_percent']:.0f}%), "
            f"RAM avg {summary['avg_ram_percent']:.0f}% (peak {summary['peak_ram_percent']:.0f}%)"
        ]

        if summary['disk_read_mb'] > 0 or summary['disk_write_mb'] > 0:
            lines.append(
                f"Disk I/O: {summary['disk_read_mb']:.1f} MB read, "
                f"{summary['disk_write_mb']:.1f} MB write"
            )

        return " | ".join(lines)

    def check_resources(self) -> tuple[bool, Optional[str]]:
        """Check if system resources are healthy.

        Returns:
            Tuple of (is_healthy, warning_message)
        """
        current = self.get_current_stats().get('current', {})

        # Check CPU
        cpu_percent = current.get('cpu_percent', 0)
        if cpu_percent > 95:
            return False, f"CPU usage critical: {cpu_percent:.0f}%"

        # Check RAM
        ram_available_gb = current.get('ram_available_gb', 0)
        if ram_available_gb < 0.5:
            return False, f"RAM critically low: {ram_available_gb:.1f} GB available"
        elif ram_available_gb < 1.0:
            return True, f"RAM running low: {ram_available_gb:.1f} GB available"

        return True, None

