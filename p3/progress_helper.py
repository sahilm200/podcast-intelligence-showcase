"""Enhanced progress bars with ETA for P3 operations."""

import time
from typing import Optional
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    MofNCompleteColumn
)
from rich.console import Console

console = Console()


class P3Progress:
    """Enhanced progress tracking with ETA for P3 operations."""

    @staticmethod
    def create_progress_bar(description: str = "Processing") -> Progress:
        """Create a rich progress bar with ETA.

        Args:
            description: Description of the operation

        Returns:
            Progress instance configured for P3 operations
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False  # Keep progress bar visible after completion
        )

    @staticmethod
    def create_spinner(description: str = "Processing") -> Progress:
        """Create a spinner for indeterminate progress.

        Args:
            description: Description of the operation

        Returns:
            Progress instance with spinner
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True  # Hide spinner after completion
        )

    @staticmethod
    def format_eta(seconds: float) -> str:
        """Format ETA in human-readable format.

        Args:
            seconds: Estimated seconds remaining

        Returns:
            Formatted ETA string (e.g., "2m 30s", "1h 15m")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h {mins}m"


class ProgressTracker:
    """Track operation progress with time estimation."""

    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker.

        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()

    def update(self, increment: int = 1) -> dict:
        """Update progress and get statistics.

        Args:
            increment: Number of items completed

        Returns:
            Dictionary with progress statistics
        """
        self.current += increment
        elapsed = time.time() - self.start_time

        # Calculate ETA
        if self.current > 0:
            avg_time_per_item = elapsed / self.current
            remaining_items = self.total - self.current
            eta_seconds = avg_time_per_item * remaining_items
        else:
            eta_seconds = 0

        return {
            'current': self.current,
            'total': self.total,
            'percent': (self.current / self.total * 100) if self.total > 0 else 0,
            'elapsed_sec': elapsed,
            'eta_sec': eta_seconds,
            'eta_formatted': P3Progress.format_eta(eta_seconds),
            'avg_time_per_item': elapsed / self.current if self.current > 0 else 0
        }

    def is_complete(self) -> bool:
        """Check if operation is complete."""
        return self.current >= self.total

    def get_summary(self) -> str:
        """Get completion summary.

        Returns:
            Formatted summary string
        """
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.current if self.current > 0 else 0

        return (f"Completed {self.current}/{self.total} items in "
                f"{P3Progress.format_eta(elapsed)} "
                f"(avg: {avg_time:.1f}s per item)")

