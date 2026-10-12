"""Dependency checking and validation for audio digest pipeline."""

import subprocess
import sys
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)


class DependencyChecker:
    """Check and validate dependencies for audio digest generation."""

    @staticmethod
    def check_python_version() -> Tuple[bool, str]:
        """Check if Python version meets requirements.

        Returns:
            Tuple of (is_compatible, message)
        """
        major, minor = sys.version_info[:2]
        current_version = f"{major}.{minor}"

        # Parakeet requires Python 3.7+
        if (major, minor) < (3, 7):
            return False, f"Python 3.7+ required, current: {current_version}"

        # TTS narration requires Python 3.9+
        if (major, minor) < (3, 9):
            return True, f"Python {current_version} (TTS narration requires 3.9+, will be disabled)"

        return True, f"Python {current_version} ✓"

    @staticmethod
    def check_ffmpeg() -> Tuple[bool, str]:
        """Check if ffmpeg is available.

        Returns:
            Tuple of (is_available, message)
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Extract version from first line
                version_line = result.stdout.split('\n')[0]
                return True, f"ffmpeg {version_line.split()[2]} ✓"
            return False, "ffmpeg command failed"
        except FileNotFoundError:
            return False, "ffmpeg not found - install from https://ffmpeg.org"
        except Exception as e:
            return False, f"ffmpeg check failed: {e}"

    @staticmethod
    def check_ollama() -> Tuple[bool, str]:
        """Check if Ollama is available and running.

        Returns:
            Tuple of (is_available, message)
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Count models
                model_lines = [l for l in result.stdout.split('\n')[1:] if l.strip()]
                return True, f"Ollama running ({len(model_lines)} models) ✓"
            return False, "Ollama not running"
        except FileNotFoundError:
            return False, "Ollama not found - install from https://ollama.com"
        except Exception as e:
            return False, f"Ollama check failed: {e}"

    @staticmethod
    def check_piper_tts() -> Tuple[bool, str]:
        """Check if Piper TTS is available.

        Returns:
            Tuple of (is_available, message)
        """
        # Check Python version first
        if sys.version_info < (3, 9):
            return False, "Requires Python 3.9+ (narration will be disabled)"

        # Method 1: Try importing piper as a Python module
        try:
            import importlib.util
            spec = importlib.util.find_spec("piper")
            if spec is not None:
                return True, "piper-tts installed ✓"
        except (ImportError, ModuleNotFoundError, ValueError):
            pass

        # Method 2: Check if piper command exists (use --help since --version requires -m flag)
        try:
            result = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # piper --help returns 0 on success
            if result.returncode == 0 and "piper" in result.stdout.lower():
                return True, "piper-tts available ✓"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Method 3: Check if piper module is available via python -m
        try:
            result = subprocess.run(
                [sys.executable, "-m", "piper", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, "piper-tts available (pip) ✓"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Not installed - provide contextual message
        if sys.version_info >= (3, 12):
            import platform
            system = platform.system()
            return False, f"piper-tts not installed (Python 3.12 on {system} - may need build tools)"
        else:
            return False, "piper-tts not installed (will auto-install on first use)"

    @staticmethod
    def check_disk_space(required_gb: float = 1.0) -> Tuple[bool, str]:
        """Check if sufficient disk space is available.

        Args:
            required_gb: Required space in GB

        Returns:
            Tuple of (is_sufficient, message)
        """
        try:
            import shutil
            stat = shutil.disk_usage(".")
            free_gb = stat.free / (1024 ** 3)

            if free_gb >= required_gb:
                return True, f"Disk space: {free_gb:.1f} GB free ✓"
            else:
                return False, f"Low disk space: {free_gb:.1f} GB free (need {required_gb} GB)"
        except Exception as e:
            return True, f"Could not check disk space: {e}"

    @staticmethod
    def run_pre_flight_checks(require_narration: bool = False) -> Dict[str, Tuple[bool, str]]:
        """Run all pre-flight checks for audio digest generation.

        Args:
            require_narration: Whether TTS narration is required

        Returns:
            Dictionary of check results {check_name: (passed, message)}
        """
        checks = {
            "Python Version": DependencyChecker.check_python_version(),
            "ffmpeg": DependencyChecker.check_ffmpeg(),
            "Ollama": DependencyChecker.check_ollama(),
            "Disk Space": DependencyChecker.check_disk_space(1.0),
        }

        # Only check piper if narration is required
        if require_narration:
            checks["Piper TTS"] = DependencyChecker.check_piper_tts()

        return checks

    @staticmethod
    def check_ollama_base_url(base_url: str) -> Tuple[bool, str]:
        """Check if Ollama base URL is reachable."""
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
            if response.status_code == 200:
                return True, f"Ollama base URL reachable ({base_url}) ✓"
            return False, f"Ollama base URL unreachable ({base_url})"
        except Exception as e:
            return False, f"Ollama base URL check failed ({base_url}): {e}"

    @staticmethod
    def ensure_dependencies(settings: dict,
                            require_narration: bool = False,
                            use_tts_api: bool = False,
                            console: Optional[object] = None) -> Dict[str, Tuple[bool, str]]:
        """Check dependencies and attempt auto-install where possible."""
        checks = DependencyChecker.run_pre_flight_checks(require_narration=require_narration and not use_tts_api)

        if require_narration and not use_tts_api:
            piper_ok, _ = checks.get("Piper TTS", (True, ""))
            if not piper_ok:
                try:
                    from .tts_generator import TTSGenerator
                    piper_voice = settings.get("audio_digest", {}).get("narration", {}).get(
                        "voice", "en_US-lessac-medium"
                    )
                    tts = TTSGenerator(use_api=False, auto_install=True, piper_voice=piper_voice)
                    checks["Piper TTS"] = (tts.is_available(), "piper-tts auto-install attempted")
                except Exception as e:
                    checks["Piper TTS"] = (False, f"piper-tts auto-install failed: {e}")

        base_url = settings.get("ollama_base_url")
        if base_url:
            ollama_ok, _ = checks.get("Ollama", (True, ""))
            if ollama_ok:
                checks["Ollama Base URL"] = DependencyChecker.check_ollama_base_url(base_url)

        # Ensure Ollama model is available when configured (best-effort)
        if checks.get("Ollama", (False, ""))[0]:
            try:
                from .model_manager import ModelManager
                model_name = settings.get("llm_model")
                if model_name:
                    ModelManager.ensure_model_available(model_name, console)
            except Exception as e:
                logger.warning(f"Failed to ensure Ollama model availability: {e}")

        return checks

    @staticmethod
    def print_dependency_status(checks: Dict[str, Tuple[bool, str]], verbose: bool = True):
        """Print dependency check results in a user-friendly format.

        Args:
            checks: Dictionary from run_pre_flight_checks()
            verbose: Whether to print all checks or just failures
        """
        print("\n" + "="*60)
        print("Dependency Status")
        print("="*60)

        all_passed = True
        critical_failed = []
        warnings = []

        for check_name, (passed, message) in checks.items():
            if verbose or not passed:
                status = "✓" if passed else "✗"
                print(f"{status} {check_name}: {message}")

            if not passed:
                all_passed = False
                if check_name in ["ffmpeg", "Ollama"]:
                    critical_failed.append(check_name)
                else:
                    warnings.append(check_name)

        print("="*60)

        if critical_failed:
            print(f"\n⚠ CRITICAL: {', '.join(critical_failed)} required for audio digest generation")
            print("Audio digest will fail without these dependencies.")
            return False
        elif warnings:
            print(f"\n⚠ WARNING: {', '.join(warnings)} not available")
            print("Some features may be disabled, but core functionality will work.")
        else:
            print("\n✓ All dependencies satisfied!")

        print()
        return all_passed

    @staticmethod
    def get_python_upgrade_instructions() -> str:
        """Get platform-specific Python upgrade instructions.

        Returns:
            Formatted instructions string
        """
        import platform
        system = platform.system().lower()

        instructions = "\nTo enable TTS narration, upgrade to Python 3.9+:\n\n"

        if system == "linux":
            instructions += """Linux:
  # Ubuntu/Debian
  sudo apt update
  sudo apt install python3.9 python3.9-venv

  # Then recreate virtual environment:
  python3.9 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
"""
        elif system == "darwin":
            instructions += """macOS:
  # Using Homebrew
  brew install python@3.9

  # Then recreate virtual environment:
  python3.9 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
"""
        elif system == "windows":
            instructions += """Windows:
  1. Download Python 3.9+ from https://www.python.org/downloads/
  2. Install and check "Add to PATH"
  3. Recreate virtual environment:
     python -m venv venv
     venv\\Scripts\\activate
     pip install -r requirements.txt
"""
        else:
            instructions += f"  Visit https://www.python.org/downloads/ for {system}\n"

        instructions += "\nAlternatively, disable TTS narration in config/feeds.yaml:\n"
        instructions += "  narration:\n"
        instructions += "    enabled: false\n"

        return instructions

