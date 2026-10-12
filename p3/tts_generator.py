"""Text-to-speech generation for audio digest narration."""

import subprocess
import logging
import sys
import time
import re
from pathlib import Path
from typing import Optional
import tempfile
import os
import httpx

logger = logging.getLogger(__name__)


def load_api_key_from_file(file_path: str) -> Optional[str]:
    """Load API key from a file.

    Args:
        file_path: Path to the API key file (relative to project root or absolute)

    Returns:
        API key string or None if file not found or empty
    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent
            path = project_root / file_path

        if path.exists() and path.is_file():
            with open(path, 'r') as f:
                api_key = f.read().strip()
                if api_key and not api_key.startswith('your-') and not api_key.startswith('gsk_your_'):
                    return api_key
        return None
    except Exception as e:
        logger.warning(f"Could not load API key from {file_path}: {e}")
        return None


class TTSGenerator:
    """Generate speech audio from text using local Piper or external API."""

    def __init__(self, voice: str = "en_US-lessac-medium", speed: float = 1.0, auto_install: bool = True,
                 use_api: bool = False, api_provider: str = None, api_model: str = None,
                 api_key: str = None, api_key_file: str = None, api_base_url: str = None,
                 piper_voice: str = "en_US-lessac-medium", max_retries: int = 3):
        """Initialize TTS generator.

        Args:
            voice: Voice to use for API TTS (provider-specific: Arista-PlayAI for Groq, alloy for OpenAI)
            speed: Speech speed multiplier (1.0 = normal)
            auto_install: Whether to auto-install piper-tts if not found
            use_api: Whether to use external API instead of local Piper
            api_provider: API provider ("groq", "openai")
            api_model: Model to use for API TTS
            api_key: API key (optional, will load from file or env)
            api_key_file: Path to API key file
            api_base_url: Override API base URL
            piper_voice: Piper voice model to use for local TTS fallback (e.g., en_US-lessac-medium)
        """
        self.voice = voice  # API voice (provider-specific)
        self.piper_voice = piper_voice  # Local Piper voice for fallback
        self.speed = speed
        self.auto_install = auto_install
        self.use_api = use_api
        self.api_provider = api_provider.lower() if api_provider else None
        self.api_model = api_model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.max_retries = max_retries
        self._piper_voice_checked = False

        # Only initialize Piper if not using API
        if not self.use_api:
            self._piper_available = self._ensure_piper_available()
        else:
            self._piper_available = False

            # Load API key from file if specified
            if not self.api_key and api_key_file:
                self.api_key = load_api_key_from_file(api_key_file)
                if self.api_key:
                    logger.info(f"Loaded TTS API key from {api_key_file}")

            # Load API key from environment if not provided
            if not self.api_key:
                if self.api_provider == "groq":
                    self.api_key = os.getenv("GROQ_API_KEY")
                elif self.api_provider == "openai":
                    self.api_key = os.getenv("OPENAI_API_KEY")

                if self.api_key:
                    logger.info("Loaded TTS API key from environment")

            # Set default API base URL
            if not self.api_base_url:
                if self.api_provider == "groq":
                    self.api_base_url = "https://api.groq.com/openai/v1"
                elif self.api_provider == "openai":
                    self.api_base_url = "https://api.openai.com/v1"

    def _check_piper_installed(self) -> bool:
        """Check if piper-tts is installed and accessible.

        Returns:
            True if piper is accessible, False otherwise
        """
        # Method 1: Try importing piper as a Python module (most reliable)
        try:
            import importlib.util
            spec = importlib.util.find_spec("piper")
            if spec is not None:
                logger.info("piper-tts detected via module import")
                return True
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
            if result.returncode == 0 and "piper" in result.stdout.lower():
                logger.info("piper-tts detected via piper command")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
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
                logger.info("piper-tts detected via python -m piper")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        logger.debug("piper-tts not detected by any method")
        return False

    def _install_piper_tts(self) -> bool:
        """Install piper-tts via pip with multiple fallback strategies.

        Returns:
            True if installation successful, False otherwise
        """
        # Check Python version - piper-tts requires 3.9+
        if sys.version_info < (3, 9):
            logger.warning(
                f"Piper TTS requires Python 3.9+, current version is {sys.version_info.major}.{sys.version_info.minor}. "
                "Narration will be disabled."
            )
            print(f"\n⚠ Warning: Piper TTS requires Python 3.9+")
            print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}")
            print(f"   Narration will be disabled.\n")
            return False

        # Warn about known Python 3.12 compatibility issues
        if sys.version_info >= (3, 12):
            import platform
            system = platform.system()
            logger.info(f"Detected Python 3.12+ on {system} - piper-tts may have compilation issues")
            print(f"\nℹ️  Note: Python 3.12 detected on {system}")
            print("   piper-tts installation may require build tools if pre-built wheels unavailable")
            print("   Installation will try multiple strategies...\n")

        logger.info("Piper TTS not found - attempting to install via pip...")
        print("="*60)
        print("📥 Installing piper-tts for audio narration...")
        print("="*60 + "\n")

        # Define installation strategies (try in order)
        strategies = [
            # Strategy 1: Standard installation (preferred)
            {
                "name": "Standard installation",
                "cmd": [sys.executable, "-m", "pip", "install", "piper-tts"],
                "timeout": 180
            },
            # Strategy 2: Try specific version (1.2.0 known to work on some systems)
            {
                "name": "Install piper-tts 1.2.0",
                "cmd": [sys.executable, "-m", "pip", "install", "piper-tts==1.2.0"],
                "timeout": 180
            },
            # Strategy 3: Install without cache (for build issues)
            {
                "name": "Install without cache",
                "cmd": [sys.executable, "-m", "pip", "install", "--no-cache-dir", "piper-tts"],
                "timeout": 300
            },
            # Strategy 4: Try installing dependencies separately
            {
                "name": "Install piper-phonemize first",
                "cmd": [sys.executable, "-m", "pip", "install", "piper-phonemize"],
                "timeout": 300
            },
        ]

        # Try each strategy
        for i, strategy in enumerate(strategies, 1):
            try:
                print(f"Attempt {i}/{len(strategies)}: {strategy['name']}...")

                process = subprocess.Popen(
                    strategy["cmd"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Show installation progress
                output_lines = []
                for line in process.stdout:
                    output_lines.append(line.rstrip())
                    # Only show important lines to avoid clutter
                    if any(keyword in line.lower() for keyword in ['downloading', 'installing', 'successfully', 'error', 'failed']):
                        print(f"  {line.rstrip()}")

                process.wait(timeout=strategy["timeout"])

                if process.returncode == 0:
                    print(f"✓ {strategy['name']} succeeded!\n")
                    logger.info(f"Piper TTS installed successfully using: {strategy['name']}")

                    # If we installed piper-phonemize separately, now install piper-tts
                    if strategy['name'] == "Install piper-phonemize first":
                        print("Installing piper-tts...")
                        final_result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "piper-tts"],
                            capture_output=True,
                            text=True,
                            timeout=180
                        )
                        if final_result.returncode == 0:
                            print("✓ piper-tts installed successfully!")
                        else:
                            continue  # Try next strategy

                    print("\n" + "="*60)
                    print("✓ Successfully installed piper-tts!")
                    print("="*60 + "\n")
                    return True
                else:
                    print(f"✗ {strategy['name']} failed (exit code {process.returncode})")
                    logger.debug(f"Strategy failed: {strategy['name']}")
                    # Try next strategy
                    continue

            except subprocess.TimeoutExpired:
                logger.warning(f"Strategy timed out: {strategy['name']}")
                print(f"✗ {strategy['name']} timed out")
                process.kill()
                continue
            except Exception as e:
                logger.warning(f"Strategy error: {strategy['name']}: {e}")
                print(f"✗ {strategy['name']} error: {e}")
                continue

        # All strategies failed
        print("\n" + "="*60)
        print("✗ Failed to install piper-tts with all strategies")
        print("="*60)

        # Provide helpful guidance
        if sys.version_info >= (3, 12):
            import platform
            system = platform.system()
            print("\n⚠️  piper-tts installation failed on Python 3.12")
            print("\nPossible solutions:")
            print("1. The piper-tts project is archived and may not support Python 3.12 fully")
            print("2. Disable narration in config/feeds.yaml:")
            print("     narration:")
            print("       enabled: false")
            print("3. Audio digest will work without narration (clips-only mode)")

            if system == "Linux":
                print("\nFor Linux users:")
                print("  Try installing build tools:")
                print("  sudo apt-get install build-essential python3-dev  # Ubuntu/Debian")
                print("  sudo dnf install gcc gcc-c++ python3-devel        # Fedora")

        logger.error("Piper TTS installation failed with all strategies")
        return False

    def _ensure_piper_available(self) -> bool:
        """Ensure piper TTS is available, installing if necessary.

        Returns:
            True if piper is available, False otherwise
        """
        # Check if already installed
        if self._check_piper_installed():
            logger.info("Piper TTS is available")
            return True

        # Not installed - attempt auto-install if enabled
        if self.auto_install:
            if self._install_piper_tts():
                # Verify installation worked
                if self._check_piper_installed():
                    return True

        logger.warning("Piper TTS not available - narration will be disabled")
        return False

    def is_available(self) -> bool:
        """Check if TTS is available.

        Returns:
            True if TTS can be used
        """
        if self.use_api:
            return bool(self.api_key and self.api_provider)
        return self._piper_available

    def _get_piper_command(self) -> list:
        """Get the piper command to use.

        Returns:
            Command list to run piper
        """
        # Try system piper first (use --help since --version requires -m flag)
        try:
            result = subprocess.run(["piper", "--help"], capture_output=True, timeout=2)
            if result.returncode == 0:
                return ["piper"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Fall back to python module
        return [sys.executable, "-m", "piper"]

    def _ensure_piper_voice(self) -> bool:
        """Attempt to download the configured Piper voice model."""
        if self._piper_voice_checked:
            return False

        self._piper_voice_checked = True
        try:
            result = subprocess.run(
                [sys.executable, "-m", "piper.download_voices", self.piper_voice],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                logger.info(f"Downloaded Piper voice model: {self.piper_voice}")
                return True
            logger.warning(f"Piper voice download failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Piper voice download failed: {e}")
        return False

    def _parse_retry_after(self, response: httpx.Response) -> Optional[int]:
        """Parse retry-after seconds from Groq rate limit responses."""
        retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

        try:
            message = response.json().get("error", {}).get("message", "")
        except (ValueError, AttributeError, TypeError):
            message = response.text

        match = re.search(r"try again in (\\d+)s", message or "", re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def generate_speech_with_api(self, text: str, output_path: str,
                                 label: Optional[str] = None) -> Optional[str]:
        """Generate speech using external API (Groq, OpenAI).

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file

        Returns:
            Path to generated audio file, or None if failed
        """
        if not self.api_key:
            logger.warning(f"No API key found for {self.api_provider} TTS")
            logger.info("Falling back to local Piper TTS")
            # Ensure piper is available for fallback
            if not self._piper_available:
                self._piper_available = self._ensure_piper_available()
            return self.generate_speech_with_piper(text, output_path)

        try:
            logger.info(f"Generating speech with {self.api_provider} API (model: {self.api_model})")

            # Ensure output path has .mp3 extension (don't double-append)
            if not output_path.endswith('.mp3'):
                mp3_path = output_path + '.mp3'
            else:
                mp3_path = output_path
            base_path = mp3_path.rsplit('.', 1)[0]
            wav_path = base_path + '.wav'

            # Prepare request based on provider
            if self.api_provider == "groq":
                # Groq uses the OpenAI-compatible TTS endpoint
                endpoint = f"{self.api_base_url}/audio/speech"

                # Prepare payload - Groq PlayAI TTS supports voice selection
                model_name = self.api_model or "canopylabs/orpheus-v1-english"
                payload = {
                    "model": model_name,
                    "input": text,
                    "response_format": "wav"
                }

                # Add voice if specified (Groq PlayAI voices: Arista-PlayAI, Atlas-PlayAI, etc.)
                if self.voice and self.voice != "default":
                    payload["voice"] = self.voice

                response = None
                for attempt in range(max(1, self.max_retries if hasattr(self, "max_retries") else 3)):
                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(
                            endpoint,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json=payload
                        )

                    if response.status_code != 429:
                        break

                    retry_after = self._parse_retry_after(response)
                    wait_seconds = retry_after if retry_after is not None else 6
                    label_suffix = f" ({label})" if label else ""
                    logger.warning(
                        f"{self.api_provider} TTS rate limited{label_suffix}, waiting {wait_seconds}s before retry"
                    )
                    time.sleep(wait_seconds)

                if response.status_code == 200:
                    # Save wav file
                    with open(wav_path, 'wb') as f:
                        f.write(response.content)

                    try:
                        subprocess.run([
                            "ffmpeg", "-i", wav_path,
                            "-ar", "44100",
                            "-ac", "1",
                            "-c:a", "libmp3lame",
                            "-q:a", "2",
                            "-y", mp3_path
                        ], check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"FFmpeg conversion failed: {e.stderr}")
                        return None
                    finally:
                        if os.path.exists(wav_path):
                            os.remove(wav_path)

                    logger.info(f"Successfully generated speech with {self.api_provider} API")
                    return mp3_path
                else:
                    retry_model = None
                    try:
                        error_payload = response.json().get("error", {})
                        if (response.status_code == 404 and error_payload.get("code") == "model_not_found"
                                and self.api_model):
                            retry_model = "canopylabs/orpheus-v1-english"
                    except (ValueError, TypeError, AttributeError):
                        if response.status_code == 404 and self.api_model:
                            retry_model = "canopylabs/orpheus-v1-english"

                    if retry_model and retry_model != model_name:
                        logger.warning(
                            f"{self.api_provider} TTS model '{model_name}' unavailable, retrying with '{retry_model}'"
                        )
                        payload["model"] = retry_model
                        with httpx.Client(timeout=60.0) as client:
                            retry_response = client.post(
                                endpoint,
                                headers={
                                    "Authorization": f"Bearer {self.api_key}",
                                    "Content-Type": "application/json"
                                },
                                json=payload
                            )
                        if retry_response.status_code == 200:
                            with open(wav_path, 'wb') as f:
                                f.write(retry_response.content)
                            try:
                                subprocess.run([
                                    "ffmpeg", "-i", wav_path,
                                    "-ar", "44100",
                                    "-ac", "1",
                                    "-c:a", "libmp3lame",
                                    "-q:a", "2",
                                    "-y", mp3_path
                                ], check=True, capture_output=True, text=True)
                            except subprocess.CalledProcessError as e:
                                logger.error(f"FFmpeg conversion failed: {e.stderr}")
                                return None
                            finally:
                                if os.path.exists(wav_path):
                                    os.remove(wav_path)
                            logger.info(f"Successfully generated speech with {self.api_provider} API")
                            return mp3_path

                        response = retry_response

                    logger.error(f"{self.api_provider} TTS API error: {response.status_code} - {response.text}")
                    if response.status_code == 429 and not self._piper_available:
                        label_suffix = f" ({label})" if label else ""
                        print(
                            f"  ⚠ {self.api_provider} TTS rate limited{label_suffix} and Piper unavailable, "
                            "skipping narration."
                        )
                        return None
                    print(f"  ⚠ {self.api_provider} TTS API error ({response.status_code}), falling back to local Piper...")
                    if not self._piper_available:
                        self._piper_available = self._ensure_piper_available()
                    return self.generate_speech_with_piper(text, output_path)

            elif self.api_provider == "openai":
                # OpenAI TTS endpoint
                endpoint = f"{self.api_base_url}/audio/speech"

                with httpx.Client(timeout=60.0) as client:
                    response = client.post(
                        endpoint,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.api_model or "tts-1",
                            "input": text,
                            "voice": self.voice or "alloy",
                            "response_format": "mp3",
                            "speed": self.speed
                        }
                    )

                if response.status_code == 200:
                    with open(mp3_path, 'wb') as f:
                        f.write(response.content)

                    logger.info(f"Successfully generated speech with OpenAI API")
                    return mp3_path
                else:
                    logger.error(f"OpenAI TTS API error: {response.status_code} - {response.text}")
                    print(f"  ⚠ OpenAI TTS API error ({response.status_code}), falling back to local Piper...")
                    if not self._piper_available:
                        self._piper_available = self._ensure_piper_available()
                    return self.generate_speech_with_piper(text, output_path)

        except Exception as e:
            logger.error(f"API TTS error: {e}")
            print(f"  ⚠ TTS API error: {e}, falling back to local Piper...")
            if not self._piper_available:
                self._piper_available = self._ensure_piper_available()
            return self.generate_speech_with_piper(text, output_path)

    def generate_speech_with_piper(self, text: str, output_path: str,
                                   label: Optional[str] = None) -> Optional[str]:
        """Generate speech using local Piper TTS.

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file

        Returns:
            Path to generated audio file, or None if failed
        """
        if not self._piper_available:
            logger.warning("Piper TTS not available - cannot generate speech")
            return None

        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None

        # Create temp wav path (strip any existing extension first to avoid double extensions)
        base_path = output_path.rsplit('.', 1)[0] if '.' in os.path.basename(output_path) else output_path
        wav_path = base_path + '.wav'

        # Ensure final output has .mp3 extension
        if not output_path.endswith('.mp3'):
            output_path = base_path + '.mp3'

        for attempt in range(2):
            try:
                # Get piper command
                piper_cmd = self._get_piper_command()

                # Generate speech with piper
                # Echo text and pipe to piper, output to WAV file
                process = subprocess.Popen(
                    piper_cmd + ["--model", self.piper_voice, "--output_file", wav_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Send text to piper
                stdout, stderr = process.communicate(input=text, timeout=30)

                if process.returncode != 0:
                    logger.error(f"Piper TTS failed: {stderr}")

                    # Check for specific error types
                    if "model" in stderr.lower() or "not found" in stderr.lower():
                        logger.error(
                            f"Voice model '{self.piper_voice}' may not be available. "
                            "Voice models are downloaded automatically on first use, but this may have failed. "
                            "Try manually downloading from: https://github.com/rhasspy/piper/releases/tag/v1.2.0"
                        )
                        print(f"\n⚠ Voice model '{self.piper_voice}' not found!")
                        print(f"   Voice models should download automatically, but this may have failed.")
                        print(f"   Try manually downloading from: https://github.com/rhasspy/piper/releases/tag/v1.2.0")
                        print(f"   Or try a different voice in config/feeds.yaml\n")
                        if attempt == 0 and self._ensure_piper_voice():
                            continue
                        self._piper_available = False

                    return None

                # Check if WAV was created
                if not os.path.exists(wav_path):
                    logger.error("Piper did not create output file")
                    return None

                # Convert WAV to MP3 with matching audio specs (44.1kHz mono)
                try:
                    subprocess.run([
                        "ffmpeg", "-i", wav_path,
                        "-ar", "44100",
                        "-ac", "1",
                        "-c:a", "libmp3lame",
                        "-q:a", "2",
                        "-y", output_path
                    ], check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg conversion failed: {e.stderr}")
                    raise

                # Clean up temp WAV file
                os.remove(wav_path)

                logger.info(f"Generated TTS audio: {output_path}")
                return output_path

            except subprocess.TimeoutExpired:
                logger.error("TTS generation timed out")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                return None
            except subprocess.CalledProcessError as e:
                logger.error(f"TTS generation failed: {e}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                return None
            except Exception as e:
                logger.error(f"Unexpected error in TTS generation: {e}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                return None

        return None

    def generate_speech(self, text: str, output_path: str,
                        label: Optional[str] = None) -> Optional[str]:
        """Generate speech audio from text.

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file

        Returns:
            Path to generated audio file, or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None

        # Choose TTS method based on configuration
        if self.use_api:
            return self.generate_speech_with_api(text, output_path, label=label)
        else:
            return self.generate_speech_with_piper(text, output_path, label=label)

    def generate_silence(self, duration_sec: float, output_path: str) -> Optional[str]:
        """Generate silent audio segment.

        Args:
            duration_sec: Duration of silence in seconds
            output_path: Path to save audio file

        Returns:
            Path to generated audio file, or None if failed
        """
        try:
            subprocess.run([
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"anullsrc=r=44100:cl=mono",
                "-t", str(duration_sec),
                "-c:a", "libmp3lame",
                "-q:a", "2",
                "-y", output_path
            ], check=True, capture_output=True)

            logger.debug(f"Generated {duration_sec}s silence: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Silence generation failed: {e}")
            return None

