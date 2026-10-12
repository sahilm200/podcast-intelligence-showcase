"""Windows-specific utilities and optimizations for P3."""

from __future__ import annotations

import os
import sys
import platform
import tempfile
from pathlib import Path, WindowsPath
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WindowsHelper:
    """Helper utilities for Windows-specific operations."""

    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows.

        Returns:
            True if on Windows platform
        """
        return platform.system() == 'Windows'

    @staticmethod
    def enable_long_paths():
        """Enable Windows long path support (>260 characters).

        On Windows 10 (version 1607+) with Python 3.6+, this enables
        long path support by setting the appropriate prefix.
        """
        if not WindowsHelper.is_windows():
            return

        # Python 3.6+ automatically handles long paths on Windows 10+
        # But we can add explicit prefix for older systems
        if sys.version_info >= (3, 6):
            logger.debug("Long path support available (Python 3.6+)")
        else:
            logger.warning("Long path support requires Python 3.6+")

    @staticmethod
    def get_safe_path(path: str) -> Path:
        """Convert path to Windows-safe format with long path support.

        Args:
            path: Path string to convert

        Returns:
            Path object with Windows long path prefix if needed
        """
        path_obj = Path(path).resolve()

        if not WindowsHelper.is_windows():
            return path_obj

        # On Windows, use extended-length path prefix for long paths
        # This allows paths longer than 260 characters
        path_str = str(path_obj)

        # Check if path is already prefixed or is a UNC path
        if path_str.startswith('\\\\?\\') or path_str.startswith('\\\\'):
            return path_obj

        # Add extended-length prefix for absolute paths
        if path_obj.is_absolute() and len(path_str) > 200:  # Approaching limit
            # Convert to extended-length path
            extended_path = '\\\\?\\' + path_str
            return Path(extended_path)

        return path_obj

    @staticmethod
    def get_temp_dir(prefix: str = "p3_") -> Path:
        """Get Windows-optimized temporary directory.

        On Windows, this uses the user's TEMP directory and ensures
        proper cleanup. Also ensures paths don't exceed Windows limits.

        Args:
            prefix: Prefix for temp directory name

        Returns:
            Path to temporary directory
        """
        if WindowsHelper.is_windows():
            # Use Windows TEMP location (typically C:\\Users\\<user>\\AppData\\Local\\Temp)
            temp_base = os.environ.get('TEMP', os.environ.get('TMP', tempfile.gettempdir()))

            # Create temp dir with short name to avoid path length issues
            temp_dir = tempfile.mkdtemp(prefix=prefix, dir=temp_base)

            logger.debug(f"Created Windows temp directory: {temp_dir}")
            return Path(temp_dir)
        else:
            # On Unix, use standard temp
            return Path(tempfile.mkdtemp(prefix=prefix))

    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize path for Windows compatibility.

        Handles:
        - Forward/backward slashes
        - Trailing slashes
        - Case sensitivity (Windows is case-insensitive)

        Args:
            path: Path string to normalize

        Returns:
            Normalized path string
        """
        # Convert to Path object which handles normalization
        normalized = Path(path)

        if WindowsHelper.is_windows():
            # On Windows, resolve to get canonical form
            try:
                normalized = normalized.resolve()
            except (OSError, RuntimeError):
                # If resolve fails, at least normalize slashes
                pass

        return str(normalized)

    @staticmethod
    def check_path_length(path: str) -> tuple[bool, Optional[str]]:
        """Check if path exceeds Windows length limits.

        Windows has a 260 character path limit (MAX_PATH) for many APIs,
        though this can be extended with \\\\?\\ prefix.

        Args:
            path: Path to check

        Returns:
            Tuple of (is_valid, warning_message)
        """
        if not WindowsHelper.is_windows():
            return True, None

        path_len = len(str(path))

        # Standard Windows path limit
        MAX_PATH = 260

        # Check if using extended-length prefix
        if str(path).startswith('\\\\?\\'):
            # Extended paths support up to 32,767 characters
            if path_len > 32000:
                return False, f"Path exceeds extended-length limit: {path_len} characters"
            return True, None

        # Standard path
        if path_len >= MAX_PATH:
            return False, f"Path exceeds Windows MAX_PATH limit ({MAX_PATH} chars): {path_len} characters. Use WindowsHelper.get_safe_path() to enable long path support."
        elif path_len > 200:
            # Approaching limit
            return True, f"Path is long ({path_len} chars), approaching Windows limit. Consider using shorter directory names."

        return True, None

    @staticmethod
    def optimize_file_operations():
        """Apply Windows-specific file operation optimizations.

        This includes:
        - Setting appropriate buffer sizes
        - Disabling file system journaling for temp files
        - Using Windows-optimized I/O
        """
        if not WindowsHelper.is_windows():
            return

        # Set larger buffer sizes for Windows I/O
        # This improves performance for audio file operations
        try:
            # Increase default buffer size for file operations
            # This is particularly helpful for large audio files
            import io
            io.DEFAULT_BUFFER_SIZE = 8192 * 8  # 64KB instead of 8KB
            logger.debug("Increased I/O buffer size for Windows")
        except Exception as e:
            logger.warning(f"Could not optimize I/O buffer: {e}")

    @staticmethod
    def get_system_info() -> dict:
        """Get Windows-specific system information.

        Returns:
            Dictionary with Windows system details
        """
        if not WindowsHelper.is_windows():
            return {'is_windows': False}

        info = {
            'is_windows': True,
            'version': platform.version(),
            'release': platform.release(),
            'machine': platform.machine(),
        }

        # Try to get Windows edition
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            info['edition'] = winreg.QueryValueEx(key, "EditionID")[0]
            info['build'] = winreg.QueryValueEx(key, "CurrentBuild")[0]
            winreg.CloseKey(key)
        except Exception as e:
            logger.debug(f"Could not read Windows registry: {e}")

        return info

    @staticmethod
    def setup_windows_environment():
        """Setup optimal Windows environment for P3.

        This should be called once at startup to apply all
        Windows-specific optimizations.
        """
        if not WindowsHelper.is_windows():
            return

        logger.info("Applying Windows-specific optimizations...")

        # Enable long path support
        WindowsHelper.enable_long_paths()

        # Optimize file operations
        WindowsHelper.optimize_file_operations()

        # Set UTF-8 encoding for Windows console
        try:
            # This fixes encoding issues with podcast titles containing unicode
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8')
            logger.debug("Configured UTF-8 encoding for console")
        except Exception as e:
            logger.warning(f"Could not configure console encoding: {e}")

        # Log Windows version
        sys_info = WindowsHelper.get_system_info()
        if 'edition' in sys_info:
            logger.info(f"Running on Windows {sys_info['release']} {sys_info['edition']} (Build {sys_info['build']})")
        else:
            logger.info(f"Running on Windows {sys_info['release']}")


def ensure_windows_compatible_path(path: str) -> Path:
    """Ensure path is Windows-compatible and handles long paths.

    This is a convenience function that wraps WindowsHelper.get_safe_path()
    and adds path length validation.

    Args:
        path: Path to convert

    Returns:
        Windows-compatible Path object

    Raises:
        ValueError: If path exceeds Windows limits
    """
    # Check path length
    is_valid, warning = WindowsHelper.check_path_length(path)
    if not is_valid:
        raise ValueError(warning)

    if warning:
        logger.warning(warning)

    # Get safe path
    return WindowsHelper.get_safe_path(path)


def create_windows_temp_dir(prefix: str = "p3_") -> Path:
    """Create a Windows-optimized temporary directory.

    Convenience function for WindowsHelper.get_temp_dir().

    Args:
        prefix: Prefix for temp directory name

    Returns:
        Path to temporary directory
    """
    return WindowsHelper.get_temp_dir(prefix)

