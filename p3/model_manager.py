"""Model management utilities for P³."""

import subprocess
import sys
from typing import List, Dict, Tuple, Optional


class ModelManager:
    """Manage Ollama and Whisper models."""

    # Recommended Ollama models
    RECOMMENDED_MODELS = {
        'llama3.2:1b': {
            'size': '1.3GB',
            'speed': 'Very Fast',
            'quality': 'Good',
            'use_case': 'Quick summaries, testing'
        },
        'llama3.2:3b': {
            'size': '2.0GB',
            'speed': 'Fast',
            'quality': 'Very Good',
            'use_case': 'Recommended for daily use'
        },
        'llama3.1:8b': {
            'size': '4.7GB',
            'speed': 'Moderate',
            'quality': 'Excellent',
            'use_case': 'Best quality summaries'
        },
        'qwen2.5:7b': {
            'size': '4.7GB',
            'speed': 'Moderate',
            'quality': 'Excellent',
            'use_case': 'Alternative to llama3.1:8b'
        }
    }

    # Whisper model info
    WHISPER_MODELS = {
        'tiny': {'quality': '90-92%', 'speed': '~1 min/hour', 'size': '~75MB'},
        'base': {'quality': '92-94%', 'speed': '~1.5 min/hour', 'size': '~150MB'},
        'small': {'quality': '94-96%', 'speed': '~2 min/hour', 'size': '~500MB'},
        'medium': {'quality': '96-98%', 'speed': '~3 min/hour', 'size': '~1.5GB'},
        'large': {'quality': '97-99%', 'speed': '~6 min/hour', 'size': '~3GB'}
    }

    @staticmethod
    def check_ollama_available() -> bool:
        """Check if Ollama is installed and running."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def get_installed_ollama_models() -> List[str]:
        """Get list of installed Ollama models."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        # Extract model name (first column)
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
        except Exception:
            return []

    @staticmethod
    def is_model_installed(model_name: str) -> bool:
        """Check if a specific Ollama model is installed."""
        installed = ModelManager.get_installed_ollama_models()
        # Check exact match or prefix match (e.g., 'llama3.2:3b' matches 'llama3.2:3b')
        return any(model_name in installed_model for installed_model in installed)

    @staticmethod
    def download_ollama_model(model_name: str, console=None) -> bool:
        """Download an Ollama model.

        Args:
            model_name: Model to download (e.g., 'llama3.2:3b')
            console: Rich console for pretty output (optional)

        Returns:
            True if successful, False otherwise
        """
        if console:
            console.print(f"\n[yellow]📥 Model '{model_name}' not found locally[/yellow]")
            console.print(f"[blue]Downloading {model_name}...[/blue]")

            # Show model info if available
            if model_name in ModelManager.RECOMMENDED_MODELS:
                info = ModelManager.RECOMMENDED_MODELS[model_name]
                console.print(f"  Size: {info['size']}")
                console.print(f"  Quality: {info['quality']}")
                console.print(f"  Use case: {info['use_case']}")

            console.print("[cyan]This may take a few minutes depending on your connection...[/cyan]\n")
        else:
            print(f"\nDownloading {model_name}...")

        try:
            # Run ollama pull with real-time output
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                bufsize=1
            )

            # Show progress
            for line in process.stdout:
                if console:
                    console.print(line.rstrip())
                else:
                    print(line.rstrip())

            process.wait()

            if process.returncode == 0:
                if console:
                    console.print(f"[green]✓ Successfully downloaded {model_name}![/green]\n")
                else:
                    print(f"\n✓ Successfully downloaded {model_name}!")
                return True
            else:
                if console:
                    console.print(f"[red]✗ Failed to download {model_name}[/red]\n")
                else:
                    print(f"\n✗ Failed to download {model_name}")
                return False

        except FileNotFoundError:
            if console:
                console.print("[red]✗ Ollama not found. Please install from https://ollama.com[/red]")
            else:
                print("✗ Ollama not found. Please install from https://ollama.com")
            return False
        except Exception as e:
            if console:
                console.print(f"[red]✗ Error downloading model: {e}[/red]")
            else:
                print(f"✗ Error downloading model: {e}")
            return False

    @staticmethod
    def ensure_model_available(model_name: str, console=None) -> bool:
        """Ensure a model is available, downloading if necessary.

        Args:
            model_name: Model to check/download
            console: Rich console for pretty output (optional)

        Returns:
            True if model is available (or was successfully downloaded)
        """
        if ModelManager.is_model_installed(model_name):
            return True

        # Model not installed, offer to download
        return ModelManager.download_ollama_model(model_name, console)

    @staticmethod
    def get_model_status() -> Dict[str, Dict]:
        """Get status of all recommended models.

        Returns:
            Dict mapping model names to their status and info
        """
        installed = ModelManager.get_installed_ollama_models()
        status = {}

        for model_name, info in ModelManager.RECOMMENDED_MODELS.items():
            is_installed = any(model_name in inst for inst in installed)
            status[model_name] = {
                **info,
                'installed': is_installed,
                'status': '✓ Installed' if is_installed else '✗ Not installed'
            }

        return status

    @staticmethod
    def get_whisper_model_info(model_name: str) -> Optional[Dict]:
        """Get information about a Whisper model."""
        return ModelManager.WHISPER_MODELS.get(model_name)

