"""System health diagnostics for P3."""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import platform

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
        return None


class SystemDoctor:
    """Comprehensive system health checker for P3."""

    def __init__(self, db, config: dict):
        """Initialize system doctor.

        Args:
            db: P3Database instance
            config: Configuration dictionary
        """
        self.db = db
        self.config = config
        self.issues = []
        self.warnings = []
        self.passed = []

    def run_full_checkup(self) -> Dict[str, any]:
        """Run comprehensive system health check.

        Returns:
            Dictionary with check results
        """
        results = {
            'system_info': self._get_system_info(),
            'python': self._check_python(),
            'dependencies': self._check_dependencies(),
            'storage': self._check_storage(),
            'memory': self._check_memory(),
            'database': self._check_database(),
            'config': self._check_config(),
            'directories': self._check_directories(),
            'models': self._check_models(),
            'issues': self.issues,
            'warnings': self.warnings,
            'passed': self.passed,
            'overall_health': 'healthy' if not self.issues else 'degraded' if not any('CRITICAL' in i for i in self.issues) else 'critical'
        }

        return results

    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor() or 'Unknown',
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'python_implementation': platform.python_implementation()
        }

    def _check_python(self) -> Dict[str, any]:
        """Check Python version and compatibility."""
        major, minor, micro = sys.version_info[:3]
        version_str = f"{major}.{minor}.{micro}"

        result = {
            'version': version_str,
            'status': 'pass',
            'details': []
        }

        # Check minimum version (3.7+)
        if (major, minor) < (3, 7):
            result['status'] = 'fail'
            self.issues.append(f"CRITICAL: Python 3.7+ required, current: {version_str}")
            result['details'].append("Upgrade to Python 3.7 or higher")
            return result

        self.passed.append(f"Python {version_str} meets minimum requirement (3.7+)")

        # Check for TTS narration support (3.9+)
        if (major, minor) < (3, 9):
            result['status'] = 'warning'
            self.warnings.append(f"Python {version_str} does not support TTS narration (requires 3.9+)")
            result['details'].append("TTS narration will be disabled")
        else:
            self.passed.append(f"Python {version_str} supports TTS narration")

        # Check for Python 3.12 specific issues
        if (major, minor) >= (3, 12):
            result['details'].append(f"Python 3.12+ detected - piper-tts may require build tools on {platform.system()}")
            self.warnings.append(f"Python 3.12 on {platform.system()} - piper-tts installation may have issues")

        return result

    def _check_dependencies(self) -> Dict[str, Dict[str, any]]:
        """Check system dependencies."""
        from .dependency_checker import DependencyChecker

        narration_enabled = self.config.get('settings', {}).get('audio_digest', {}).get('narration', {}).get('enabled', True)
        checks = DependencyChecker.run_pre_flight_checks(require_narration=narration_enabled)

        result = {}
        for check_name, (passed, message) in checks.items():
            result[check_name] = {
                'status': 'pass' if passed else 'fail',
                'message': message
            }

            if not passed:
                if check_name in ["ffmpeg", "Ollama"]:
                    self.issues.append(f"CRITICAL: {check_name} not available - {message}")
                else:
                    self.warnings.append(f"{check_name} not available - {message}")
            else:
                self.passed.append(f"{check_name}: {message}")

        return result

    def _check_storage(self) -> Dict[str, any]:
        """Check disk storage."""
        try:
            stat = shutil.disk_usage(".")
            total_gb = stat.total / (1024 ** 3)
            used_gb = stat.used / (1024 ** 3)
            free_gb = stat.free / (1024 ** 3)
            used_percent = (stat.used / stat.total) * 100

            result = {
                'total_gb': round(total_gb, 2),
                'used_gb': round(used_gb, 2),
                'free_gb': round(free_gb, 2),
                'used_percent': round(used_percent, 1),
                'status': 'pass'
            }

            # Check for low disk space
            if free_gb < 1:
                result['status'] = 'fail'
                self.issues.append(f"CRITICAL: Low disk space - only {free_gb:.1f} GB free")
            elif free_gb < 5:
                result['status'] = 'warning'
                self.warnings.append(f"Low disk space warning - {free_gb:.1f} GB free (recommend 5+ GB)")
            else:
                self.passed.append(f"Disk space: {free_gb:.1f} GB free")

            # Check audio directory size
            audio_dir = Path("data/audio")
            if audio_dir.exists():
                audio_size = sum(f.stat().st_size for f in audio_dir.rglob('*') if f.is_file())
                audio_size_mb = audio_size / (1024 ** 2)
                result['audio_dir_mb'] = round(audio_size_mb, 2)

                if audio_size_mb > 1000:  # > 1GB
                    self.warnings.append(f"Audio directory is large ({audio_size_mb:.0f} MB) - consider running 'p3 cleanup'")

            return result

        except Exception as e:
            logger.error(f"Storage check failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def _check_memory(self) -> Dict[str, any]:
        """Check system memory."""
        psutil = _import_psutil()
        if not psutil:
            logger.warning("psutil not installed - memory check skipped")
            self.warnings.append("psutil not installed - cannot check memory (run: pip install psutil)")
            return {
                'status': 'warning',
                'message': 'psutil not installed - memory check skipped'
            }

        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            available_gb = mem.available / (1024 ** 3)
            used_percent = mem.percent

            result = {
                'total_gb': round(total_gb, 2),
                'available_gb': round(available_gb, 2),
                'used_percent': round(used_percent, 1),
                'status': 'pass'
            }

            # Check for low memory
            if available_gb < 1:
                result['status'] = 'fail'
                self.issues.append(f"CRITICAL: Low memory - only {available_gb:.1f} GB available")
            elif available_gb < 2:
                result['status'] = 'warning'
                self.warnings.append(f"Low memory warning - {available_gb:.1f} GB available (recommend 2+ GB free)")
            else:
                self.passed.append(f"Memory: {available_gb:.1f} GB available")

            return result

        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def _check_database(self) -> Dict[str, any]:
        """Check database health."""
        result = {
            'status': 'pass',
            'episode_count': 0,
            'summary_count': 0,
            'details': []
        }

        try:
            # Check if database is accessible
            episodes = self.db.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            result['episode_count'] = episodes

            summaries = self.db.conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            result['summary_count'] = summaries

            self.passed.append(f"Database accessible: {episodes} episodes, {summaries} summaries")

            # Check for pending work
            downloaded = len(self.db.get_episodes_by_status('downloaded'))
            transcribed = len(self.db.get_episodes_by_status('transcribed'))

            result['pending_transcription'] = downloaded
            result['pending_digest'] = transcribed

            if downloaded > 0:
                result['details'].append(f"{downloaded} episodes pending transcription")
            if transcribed > 0:
                result['details'].append(f"{transcribed} episodes pending digest generation")

            return result

        except Exception as e:
            logger.error(f"Database check failed: {e}")
            result['status'] = 'fail'
            result['error'] = str(e)
            self.issues.append(f"CRITICAL: Database check failed - {e}")
            return result

    def _check_config(self) -> Dict[str, any]:
        """Validate configuration file."""
        result = {
            'status': 'pass',
            'details': []
        }

        # Check if feeds are configured
        feeds = self.config.get('feeds', [])
        if not feeds:
            result['status'] = 'warning'
            self.warnings.append("No podcast feeds configured in config/feeds.yaml")
            result['details'].append("Add podcast RSS feeds to start processing")
        else:
            result['feed_count'] = len(feeds)
            self.passed.append(f"Configuration: {len(feeds)} podcast feeds configured")

        # Check settings
        settings = self.config.get('settings', {})

        # Check LLM configuration
        llm_provider = settings.get('llm_provider', 'ollama')
        llm_model = settings.get('llm_model', 'llama3.2:3b')
        result['llm_provider'] = llm_provider
        result['llm_model'] = llm_model

        # Check audio digest settings
        audio_digest = settings.get('audio_digest', {})
        result['audio_digest_enabled'] = audio_digest.get('enabled', True)
        result['narration_enabled'] = audio_digest.get('narration', {}).get('enabled', True)

        return result

    def _check_directories(self) -> Dict[str, any]:
        """Check required directories."""
        required_dirs = [
            'data',
            'data/audio',
            'outputs',
            'outputs/digests',
            'outputs/audio_digests',
            'outputs/blogs',
            'outputs/tweets',
            'outputs/thesis',
            'outputs/transcripts',
            'config'
        ]

        result = {
            'status': 'pass',
            'missing': [],
            'present': []
        }

        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                result['present'].append(dir_path)
            else:
                result['missing'].append(dir_path)

        if result['missing']:
            result['status'] = 'warning'
            self.warnings.append(f"Missing directories: {', '.join(result['missing'])} - run 'p3 init' to create")
        else:
            self.passed.append("All required directories present")

        return result

    def _check_models(self) -> Dict[str, any]:
        """Check Ollama models availability."""
        from .model_manager import ModelManager

        result = {
            'status': 'pass',
            'ollama_available': False,
            'installed_models': [],
            'missing_recommended': []
        }

        # Check if Ollama is available
        if not ModelManager.check_ollama_available():
            result['status'] = 'fail'
            self.issues.append("CRITICAL: Ollama not running - required for LLM operations")
            return result

        result['ollama_available'] = True

        # Get installed models
        installed = ModelManager.get_installed_ollama_models()
        result['installed_models'] = installed

        # Check for recommended models
        recommended = ['llama3.2:1b', 'llama3.2:3b', 'llama3.1:8b']
        configured_model = self.config.get('settings', {}).get('llm_model', 'llama3.2:3b')

        if configured_model not in installed:
            result['status'] = 'warning'
            self.warnings.append(f"Configured model '{configured_model}' not installed - will auto-download on first use")
            result['missing_recommended'].append(configured_model)
        else:
            self.passed.append(f"LLM model '{configured_model}' installed and ready")

        # Check for at least one recommended model
        has_recommended = any(model in installed for model in recommended)
        if not has_recommended:
            self.warnings.append("No recommended LLM models installed - will auto-download on first use")

        result['model_count'] = len(installed)

        return result

    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on check results."""
        recommendations = []

        # Critical issues first
        if any('ffmpeg' in issue for issue in self.issues):
            system = platform.system()
            if system == 'Linux':
                recommendations.append("Install ffmpeg: sudo apt install ffmpeg")
            elif system == 'Darwin':
                recommendations.append("Install ffmpeg: brew install ffmpeg")
            elif system == 'Windows':
                recommendations.append("Install ffmpeg: Download from https://ffmpeg.org/download.html")

        if any('Ollama' in issue for issue in self.issues):
            recommendations.append("Install Ollama: https://ollama.com")
            recommendations.append("Then run: ollama pull llama3.2:3b")

        if any('disk space' in issue.lower() for issue in self.issues):
            recommendations.append("Free up disk space or run: p3 cleanup")

        if any('memory' in issue.lower() for issue in self.issues):
            recommendations.append("Close other applications to free up memory")

        # Warnings and suggestions
        if any('feeds' in warning.lower() for warning in self.warnings):
            recommendations.append("Configure podcast feeds in config/feeds.yaml")

        if any('directories' in warning.lower() for warning in self.warnings):
            recommendations.append("Run: p3 init (to create missing directories)")

        if any('audio directory' in warning.lower() for warning in self.warnings):
            recommendations.append("Clean up old audio files: p3 cleanup")

        if any('tts' in warning.lower() or 'narration' in warning.lower() for warning in self.warnings):
            if sys.version_info < (3, 9):
                recommendations.append("Upgrade to Python 3.9+ to enable TTS narration")
                recommendations.append("Or disable narration in config/feeds.yaml (narration.enabled: false)")
            elif sys.version_info >= (3, 12):
                recommendations.append("piper-tts may need manual installation on Python 3.12")
                recommendations.append("Try: pip install piper-tts")
                recommendations.append("Or disable narration in config/feeds.yaml")

        if any('model' in warning.lower() for warning in self.warnings):
            recommendations.append("Models will auto-download on first use")
            recommendations.append("Or manually install: ollama pull llama3.2:3b")

        # General recommendations
        if not recommendations and not self.issues:
            recommendations.append("System is healthy! Ready to run: p3 auto")

        return recommendations

