"""Audio transcription using Whisper and Parakeet."""

import json
import os
import re
import subprocess
import tempfile
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any
import whisper
import httpx

from .database import P3Database
from .audio_processor import AudioProcessor


def parse_rate_limit_wait_time(error_message: str) -> Optional[int]:
    """Parse the wait time from a rate limit error message.

    Extracts wait time from messages like "Please try again in 6m18s" or "try again in 45s"

    Args:
        error_message: Error message from API

    Returns:
        Wait time in seconds, or None if not parseable
    """
    # Try to match "Xm Ys" format
    match = re.search(r'try again in (\d+)m\s*(\d+(?:\.\d+)?)?s', error_message, re.IGNORECASE)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2)) if match.group(2) else 0
        return int(minutes * 60 + seconds)

    # Try to match "Xs" format
    match = re.search(r'try again in (\d+(?:\.\d+)?)s', error_message, re.IGNORECASE)
    if match:
        return int(float(match.group(1)))

    return None

# Optional Parakeet MLX support
try:
    from parakeet_mlx import from_pretrained as parakeet_from_pretrained
    PARAKEET_AVAILABLE = True
except ImportError:
    PARAKEET_AVAILABLE = False

# Optional faster-whisper support (4x faster, more efficient)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


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
        print(f"Warning: Could not load API key from {file_path}: {e}")
        return None


class AudioTranscriber:
    def __init__(self, db: P3Database, whisper_model: str = "base",
                 use_parakeet: bool = False, parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v2",
                 use_faster_whisper: bool = False,
                 use_api: bool = False, api_provider: str = None,
                 api_model: str = None, api_key: str = None,
                 api_key_file: str = None, api_base_url: str = None,
                 wait_for_ratelimit: bool = False, max_wait_retries: int = 3,
                 api_fallback_model: str = None,
                 api_fallback_models: Optional[List[str]] = None):
        self.db = db
        self.whisper_model = whisper_model
        self.use_parakeet = use_parakeet
        self.parakeet_model = parakeet_model
        self.use_faster_whisper = use_faster_whisper
        self.use_api = use_api
        self.api_provider = api_provider.lower() if api_provider else None
        self.api_model = api_model
        self.api_fallback_model = api_fallback_model
        if isinstance(api_fallback_models, str):
            api_fallback_models = [api_fallback_models]
        fallback_models = list(api_fallback_models or [])
        if api_fallback_model:
            fallback_models.append(api_fallback_model)
        self.api_fallback_models = [
            model for model in fallback_models if model and model != api_model
        ]
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.wait_for_ratelimit = wait_for_ratelimit
        self.max_wait_retries = max_wait_retries
        self.whisper = None
        self.parakeet = None
        self.faster_whisper = None
        self._model_load_lock = threading.Lock()

        # Load API key from file if using API transcription
        if self.use_api and not self.api_key and api_key_file:
            self.api_key = load_api_key_from_file(api_key_file)
            if self.api_key:
                print(f"Loaded transcription API key from {api_key_file}")

        # Load API key from environment if not provided
        if self.use_api and not self.api_key:
            if self.api_provider == "groq":
                self.api_key = os.getenv("GROQ_API_KEY")
            elif self.api_provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.api_provider == "assemblyai":
                self.api_key = os.getenv("ASSEMBLYAI_API_KEY")

            if self.api_key:
                print(f"Loaded transcription API key from environment")

        # Load API key from environment if not provided (extended provider list)
        if self.use_api and not self.api_key:
            if self.api_provider == "deepgram":
                self.api_key = os.getenv("DEEPGRAM_API_KEY")

        # Set default API base URL
        if self.use_api and not self.api_base_url:
            if self.api_provider == "groq":
                self.api_base_url = "https://api.groq.com/openai/v1"
            elif self.api_provider == "openai":
                self.api_base_url = "https://api.openai.com/v1"
            elif self.api_provider == "assemblyai":
                self.api_base_url = "https://api.assemblyai.com/v2"
            elif self.api_provider == "deepgram":
                self.api_base_url = "https://api.deepgram.com/v1"
        
    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self.whisper is None:
            print(f"Loading Whisper model: {self.whisper_model}")
            self.whisper = whisper.load_model(self.whisper_model)

    def _load_parakeet(self):
        """Lazy load Parakeet MLX model."""
        if self.parakeet is None and PARAKEET_AVAILABLE:
            print(f"Loading Parakeet model: {self.parakeet_model}")
            self.parakeet = parakeet_from_pretrained(self.parakeet_model)

    def _load_faster_whisper(self):
        """Lazy load faster-whisper model (thread-safe)."""
        if self.faster_whisper is not None:
            return
        with self._model_load_lock:
            if self.faster_whisper is not None:
                return  # Another thread loaded it while we waited
            if not FASTER_WHISPER_AVAILABLE:
                return
            import os
            import sys

            # Boost process priority on Windows for faster transcription
            if sys.platform == 'win32':
                try:
                    import psutil
                    p = psutil.Process()
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                    print("Boosted process priority to HIGH")
                except (ImportError, Exception):
                    pass  # Skip if psutil not available

            # Use all available CPU threads for maximum performance
            cpu_threads = os.cpu_count() or 4
            print(f"Loading faster-whisper model: {self.whisper_model} (using {cpu_threads} CPU threads)")
            # Use CPU with optimal settings for better performance
            self.faster_whisper = WhisperModel(
                self.whisper_model,
                device="cpu",
                compute_type="int8",  # Quantized for speed without major quality loss
                cpu_threads=cpu_threads,  # Use all available CPU threads
                num_workers=1  # Number of parallel workers
            )

    def transcribe_with_faster_whisper(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using faster-whisper (4x faster, more memory efficient).

        faster-whisper uses CTranslate2 for optimized CPU inference.
        Ideal for Windows systems without Apple Silicon or CUDA.
        """
        if not FASTER_WHISPER_AVAILABLE:
            print("faster-whisper not available, falling back to standard Whisper")
            return self.transcribe_with_whisper(audio_path)

        self._load_faster_whisper()

        try:
            # Transcribe with word-level timestamps and performance optimizations
            segments, info = self.faster_whisper.transcribe(
                audio_path,
                beam_size=1,  # Faster decoding (greedy search instead of beam search)
                word_timestamps=True,
                vad_filter=True,  # Voice activity detection for better accuracy
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Convert faster-whisper output to our format
            result_segments = []
            full_text = []

            import time
            start_time = time.time()
            print(f"Processing segments (audio duration: {info.duration:.1f}s / {info.duration/60:.1f}m)", flush=True)
            segment_count = 0
            last_timestamp = 0

            for segment in segments:
                result_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'speaker': None,
                    'confidence': segment.avg_logprob  # Log probability as confidence
                })
                full_text.append(segment.text.strip())
                segment_count += 1
                last_timestamp = segment.end

                # Show progress every 10 segments with timestamp and percentage
                if segment_count % 10 == 0:
                    elapsed = time.time() - start_time
                    progress_pct = (last_timestamp / info.duration * 100) if info.duration > 0 else 0
                    print(f"  [{elapsed:.0f}s] {segment_count} segments | {last_timestamp/60:.1f}m / {info.duration/60:.1f}m ({progress_pct:.0f}%)", flush=True)

            elapsed = time.time() - start_time
            print(f"✓ Completed {segment_count} segments in {elapsed:.0f}s", flush=True)

            return {
                'segments': result_segments,
                'language': info.language,
                'text': ' '.join(full_text),
                'provider': 'faster-whisper',
                'language_probability': info.language_probability
            }

        except Exception as e:
            print(f"faster-whisper transcription failed: {e}")
            print("Falling back to standard Whisper")
            return self.transcribe_with_whisper(audio_path)

    def transcribe_with_whisper(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using OpenAI Whisper."""
        self._load_whisper()
        
        try:
            result = self.whisper.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )
            
            # Convert Whisper output to our format
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'text': segment.get('text', '').strip(),
                    'speaker': None,  # Whisper doesn't do speaker detection
                    'confidence': segment.get('no_speech_prob', 0.0)
                })
            
            return {
                'segments': segments,
                'language': result.get('language'),
                'text': result.get('text', ''),
                'provider': 'whisper'
            }
            
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            return None

    def transcribe_with_parakeet(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Nvidia Parakeet MLX."""
        if not PARAKEET_AVAILABLE:
            print("Parakeet MLX not available, falling back to Whisper")
            return self.transcribe_with_whisper(audio_path)
        
        self._load_parakeet()
        
        try:
            result = self.parakeet.transcribe(audio_path)
            
            # Convert Parakeet output to our format
            segments = []
            for sentence in result.sentences:
                segments.append({
                    'start': sentence.start,
                    'end': sentence.end,
                    'text': sentence.text.strip(),
                    'speaker': None,  # Parakeet doesn't do speaker identification
                    'confidence': 1.0  # Parakeet doesn't provide confidence scores
                })
            
            return {
                'segments': segments,
                'language': 'en',  # Parakeet is English-only
                'text': result.text,
                'provider': 'parakeet-mlx'
            }
            
        except Exception as e:
            print(f"Parakeet transcription failed: {e}")
            print("Falling back to Whisper")
            return self.transcribe_with_whisper(audio_path)

    def transcribe_with_assemblyai(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio using AssemblyAI with speaker diarization.

        AssemblyAI provides:
        - High-accuracy transcription
        - Built-in speaker diarization (speaker_labels=true)
        - 100 free hours/month
        - No file size limit (handles large files natively)

        Returns segments with speaker labels (e.g., "A", "B", "C").
        """
        if not self.api_key:
            print("Warning: No AssemblyAI API key found")
            print("Falling back to local Whisper")
            return self.transcribe_with_faster_whisper(audio_path) if self.use_faster_whisper else self.transcribe_with_whisper(audio_path)

        audio_file_path = Path(audio_path)
        if not audio_file_path.exists():
            print(f"Audio file not found: {audio_path}")
            return None

        try:
            print(f"Transcribing with AssemblyAI (speaker diarization enabled)...", flush=True)

            headers = {
                "authorization": self.api_key,
                "content-type": "application/json"
            }

            # Step 1: Upload the audio file
            print("  Uploading audio file...", flush=True)
            with open(audio_path, 'rb') as audio_file:
                with httpx.Client(timeout=300.0) as client:
                    upload_response = client.post(
                        f"{self.api_base_url}/upload",
                        headers={"authorization": self.api_key},
                        content=audio_file.read()
                    )

            if upload_response.status_code != 200:
                raise Exception(f"Upload failed: {upload_response.status_code} - {upload_response.text}")

            audio_url = upload_response.json().get("upload_url")
            if not audio_url:
                raise Exception("No upload URL returned from AssemblyAI")

            # Step 2: Request transcription with speaker diarization
            print("  Starting transcription with speaker diarization...", flush=True)
            transcript_request = {
                "audio_url": audio_url,
                "speaker_labels": True,  # Enable speaker diarization
                "speakers_expected": None  # Let AssemblyAI auto-detect speaker count
            }

            with httpx.Client(timeout=60.0) as client:
                transcript_response = client.post(
                    f"{self.api_base_url}/transcript",
                    headers=headers,
                    json=transcript_request
                )

            if transcript_response.status_code != 200:
                raise Exception(f"Transcription request failed: {transcript_response.status_code} - {transcript_response.text}")

            transcript_id = transcript_response.json().get("id")
            if not transcript_id:
                raise Exception("No transcript ID returned from AssemblyAI")

            # Step 3: Poll for completion
            poll_url = f"{self.api_base_url}/transcript/{transcript_id}"
            max_polls = 120  # 10 minutes max (5s intervals)
            poll_count = 0

            while poll_count < max_polls:
                with httpx.Client(timeout=30.0) as client:
                    poll_response = client.get(poll_url, headers=headers)

                if poll_response.status_code != 200:
                    raise Exception(f"Poll failed: {poll_response.status_code} - {poll_response.text}")

                result = poll_response.json()
                status = result.get("status")

                if status == "completed":
                    break
                elif status == "error":
                    error_msg = result.get("error", "Unknown error")
                    raise Exception(f"Transcription failed: {error_msg}")
                elif status in ("queued", "processing"):
                    poll_count += 1
                    if poll_count % 6 == 0:  # Log every 30s
                        print(f"  Still processing... ({poll_count * 5}s elapsed)", flush=True)
                    time.sleep(5)
                else:
                    raise Exception(f"Unknown status: {status}")
            else:
                raise Exception("Transcription timed out after 10 minutes")

            # Step 4: Parse results with speaker labels
            utterances = result.get("utterances", [])
            words = result.get("words", [])

            # Build segments from utterances (speaker-grouped chunks)
            segments = []
            if utterances:
                # Use utterances for speaker-aware segments
                for utterance in utterances:
                    segments.append({
                        'start': utterance.get('start', 0) / 1000.0,  # Convert ms to seconds
                        'end': utterance.get('end', 0) / 1000.0,
                        'text': utterance.get('text', '').strip(),
                        'speaker': utterance.get('speaker'),  # e.g., "A", "B", "C"
                        'confidence': utterance.get('confidence', 1.0)
                    })
            elif words:
                # Fallback: group words into segments by speaker
                current_segment = None
                for word in words:
                    speaker = word.get('speaker')
                    if current_segment is None or current_segment['speaker'] != speaker:
                        if current_segment:
                            segments.append(current_segment)
                        current_segment = {
                            'start': word.get('start', 0) / 1000.0,
                            'end': word.get('end', 0) / 1000.0,
                            'text': word.get('text', ''),
                            'speaker': speaker,
                            'confidence': word.get('confidence', 1.0)
                        }
                    else:
                        current_segment['end'] = word.get('end', 0) / 1000.0
                        current_segment['text'] += ' ' + word.get('text', '')
                if current_segment:
                    segments.append(current_segment)
            else:
                # Last resort: single segment with full text
                segments.append({
                    'start': 0,
                    'end': result.get('audio_duration', 0),
                    'text': result.get('text', '').strip(),
                    'speaker': None,
                    'confidence': 1.0
                })

            # Count unique speakers
            unique_speakers = set(s.get('speaker') for s in segments if s.get('speaker'))
            speaker_info = f", {len(unique_speakers)} speakers detected" if unique_speakers else ""

            print(f"✓ Transcribed with AssemblyAI ({len(segments)} segments{speaker_info})", flush=True)

            return {
                'segments': segments,
                'language': result.get('language_code', 'en'),
                'text': result.get('text', ''),
                'provider': 'assemblyai',
                'duration': result.get('audio_duration', 0),
                'speaker_count': len(unique_speakers)
            }

        except Exception as e:
            print(f"AssemblyAI transcription failed: {e}")
            print("Falling back to local Whisper")
            return self.transcribe_with_faster_whisper(audio_path) if self.use_faster_whisper else self.transcribe_with_whisper(audio_path)

    def transcribe_with_deepgram(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Deepgram Nova-2 (12,000 free minutes/month).

        Deepgram provides:
        - Fast, accurate transcription with Nova-2 model
        - Speaker diarization built-in
        - 12,000 free minutes per month (no credit card needed)
        - Handles large files natively

        Returns segments with timestamps and optional speaker labels.
        """
        if not self.api_key:
            print("Warning: No Deepgram API key found, falling back to local Whisper")
            return self.transcribe_with_faster_whisper(audio_path) if self.use_faster_whisper else self.transcribe_with_whisper(audio_path)

        audio_file_path = Path(audio_path)
        if not audio_file_path.exists():
            print(f"Audio file not found: {audio_path}")
            return None

        try:
            print(f"Transcribing with Deepgram Nova-2 (speaker diarization enabled)...", flush=True)

            model = self.api_model or "nova-2"
            url = f"{self.api_base_url}/listen?model={model}&diarize=true&punctuate=true&paragraphs=true&utterances=true"

            with open(audio_path, 'rb') as audio_file:
                with httpx.Client(timeout=600.0) as client:
                    response = client.post(
                        url,
                        headers={
                            "Authorization": f"Token {self.api_key}",
                            "Content-Type": "audio/mpeg",
                        },
                        content=audio_file.read(),
                    )

            if response.status_code != 200:
                raise Exception(f"Deepgram API error {response.status_code}: {response.text}")

            result = response.json()
            channel = result.get("results", {}).get("channels", [{}])[0]
            alternatives = channel.get("alternatives", [{}])
            if not alternatives:
                raise Exception("No transcription alternatives returned by Deepgram")

            best = alternatives[0]
            utterances = result.get("results", {}).get("utterances", [])

            segments = []
            if utterances:
                for utterance in utterances:
                    segments.append({
                        'start': utterance.get('start', 0),
                        'end': utterance.get('end', 0),
                        'text': utterance.get('transcript', '').strip(),
                        'speaker': f"Speaker_{utterance.get('speaker', 0)}",
                        'confidence': utterance.get('confidence', 1.0),
                    })
            else:
                # Fallback: use words grouped into chunks
                words = best.get("words", [])
                current_seg: Optional[dict] = None
                for word in words:
                    if current_seg is None:
                        current_seg = {
                            'start': word.get('start', 0),
                            'end': word.get('end', 0),
                            'text': word.get('word', ''),
                            'speaker': f"Speaker_{word.get('speaker', 0)}",
                            'confidence': word.get('confidence', 1.0),
                        }
                    elif word.get('speaker') != int(current_seg['speaker'].split('_')[-1]):
                        segments.append(current_seg)
                        current_seg = {
                            'start': word.get('start', 0),
                            'end': word.get('end', 0),
                            'text': word.get('word', ''),
                            'speaker': f"Speaker_{word.get('speaker', 0)}",
                            'confidence': word.get('confidence', 1.0),
                        }
                    else:
                        current_seg['end'] = word.get('end', 0)
                        current_seg['text'] += ' ' + word.get('word', '')
                if current_seg:
                    segments.append(current_seg)

            if not segments:
                segments.append({
                    'start': 0,
                    'end': result.get('metadata', {}).get('duration', 0),
                    'text': best.get('transcript', '').strip(),
                    'speaker': None,
                    'confidence': best.get('confidence', 1.0),
                })

            unique_speakers = set(s.get('speaker') for s in segments if s.get('speaker'))
            speaker_info = f", {len(unique_speakers)} speakers detected" if unique_speakers else ""
            print(f"✓ Transcribed with Deepgram ({len(segments)} segments{speaker_info})", flush=True)

            return {
                'segments': segments,
                'language': result.get('results', {}).get('channels', [{}])[0].get('detected_language', 'en'),
                'text': best.get('transcript', ''),
                'provider': 'deepgram',
                'duration': result.get('metadata', {}).get('duration', 0),
                'speaker_count': len(unique_speakers),
            }

        except Exception as e:
            print(f"Deepgram transcription failed: {e}")
            print("Falling back to local Whisper")
            return self.transcribe_with_faster_whisper(audio_path) if self.use_faster_whisper else self.transcribe_with_whisper(audio_path)

    def transcribe_with_api(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio using external API (Groq, OpenAI, AssemblyAI).

        This is ideal for:
        - Groq: Free tier available, ultra-fast transcription
        - OpenAI: Pay-per-use, high accuracy
        - AssemblyAI: Speaker diarization included, 100 free hours/month

        Saves local compute resources while maintaining quality.
        Automatically handles files over 25MB by splitting into chunks.
        """
        if not self.api_key:
            print(f"Warning: No API key found for {self.api_provider} transcription")
            print("Falling back to local Whisper")
            return self.transcribe_with_faster_whisper(audio_path) if self.use_faster_whisper else self.transcribe_with_whisper(audio_path)

        # Route to specialized providers
        if self.api_provider == "assemblyai":
            return self.transcribe_with_assemblyai(audio_path)
        if self.api_provider == "deepgram":
            return self.transcribe_with_deepgram(audio_path)

        try:
            audio_file_path = Path(audio_path)
            if not audio_file_path.exists():
                print(f"Audio file not found: {audio_path}")
                return None

            # Check file size - Groq/OpenAI limit is 25MB
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            max_size_mb = 25.0

            if file_size_mb > max_size_mb:
                print(f"File size ({file_size_mb:.1f}MB) exceeds {max_size_mb}MB limit, using chunked transcription...")
                return self._transcribe_chunked(audio_path, max_size_mb)

            # File is under limit, transcribe directly
            try:
                return self._transcribe_single_file(audio_path)
            except Exception as e:
                if self._is_rate_limit_error(e):
                    print(f"  ⚠ API rate limited, retrying with fallback model...", flush=True)
                    fallback_result = self._retry_with_fallback_models(audio_path)
                    if fallback_result:
                        return fallback_result
                raise

        except Exception as e:
            print(f"API transcription error: {e}")
            print("Falling back to local Whisper")
            return self.transcribe_with_faster_whisper(audio_path) if self.use_faster_whisper else self.transcribe_with_whisper(audio_path)

    def _transcribe_single_file(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe a single audio file under the size limit.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription result or None
        """
        audio_file_path = Path(audio_path)
        print(f"Transcribing with {self.api_provider} API (model: {self.api_model})...", flush=True)

        retries = 0
        max_retries = self.max_wait_retries if self.wait_for_ratelimit else 0

        while True:
            with open(audio_path, 'rb') as audio_file:
                with httpx.Client(timeout=300.0) as client:  # 5 minute timeout
                    response = client.post(
                        f"{self.api_base_url}/audio/transcriptions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}"
                        },
                        files={
                            "file": (audio_file_path.name, audio_file, "audio/mpeg")
                        },
                        data={
                            "model": self.api_model,
                            "response_format": "verbose_json",
                            "timestamp_granularities[]": "segment"
                        }
                    )

            if response.status_code == 200:
                result = response.json()

                segments = []
                for segment in result.get('segments', []):
                    segments.append({
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'text': segment.get('text', '').strip(),
                        'speaker': None,
                        'confidence': 1.0
                    })

                if not segments and result.get('text'):
                    segments.append({
                        'start': 0,
                        'end': result.get('duration', 0),
                        'text': result.get('text', '').strip(),
                        'speaker': None,
                        'confidence': 1.0
                    })

                print(f"✓ Transcribed with {self.api_provider} API ({len(segments)} segments)", flush=True)

                return {
                    'segments': segments,
                    'language': result.get('language', 'en'),
                    'text': result.get('text', ''),
                    'provider': f'{self.api_provider}-api',
                    'duration': result.get('duration', 0)
                }
            elif response.status_code == 429 and self.wait_for_ratelimit and retries < max_retries:
                # Rate limited - parse wait time and retry
                wait_seconds = parse_rate_limit_wait_time(response.text)
                if wait_seconds:
                    wait_seconds = min(wait_seconds + 5, 600)  # Add 5s buffer, max 10 min
                    print(f"  ⏳ Rate limited. Waiting {wait_seconds}s before retry ({retries + 1}/{max_retries})...")
                    time.sleep(wait_seconds)
                    retries += 1
                    continue
                else:
                    raise Exception(f"API returned {response.status_code}: {response.text}")
            else:
                raise Exception(f"API returned {response.status_code}: {response.text}")

    def _transcribe_chunk_locally(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe a single chunk using local faster-whisper or whisper.

        Used as fallback when API transcription fails for a chunk.

        Args:
            audio_path: Path to audio chunk file

        Returns:
            Transcription result or None
        """
        if self.use_faster_whisper or FASTER_WHISPER_AVAILABLE:
            return self.transcribe_with_faster_whisper(audio_path)
        else:
            return self.transcribe_with_whisper(audio_path)

    def _transcribe_chunked(self, audio_path: str, max_size_mb: float = 20.0) -> Optional[Dict[str, Any]]:
        """Transcribe large audio file by splitting into chunks with parallel processing.

        Uses a bidirectional strategy: API transcription works from the front
        while local transcription works from the back. This maximizes throughput
        by utilizing both API bandwidth and local CPU concurrently.

        Args:
            audio_path: Path to audio file
            max_size_mb: Maximum chunk size (20MB for safety margin below 25MB limit)

        Returns:
            Merged transcription result or None
        """
        audio_processor = AudioProcessor()
        temp_dir = None

        try:
            # Create temp directory for chunks
            temp_dir = tempfile.mkdtemp(prefix="p3_chunks_")
            print(f"Splitting audio into chunks (max {max_size_mb}MB each)...")

            # Split audio into chunks
            chunks = audio_processor.split_audio_into_chunks(
                audio_path, temp_dir, max_size_mb=max_size_mb, overlap_seconds=0.5
            )

            num_chunks = len(chunks)
            print(f"Created {num_chunks} chunks, transcribing in parallel (API→ + ←local)...")

            # Shared state for parallel processing
            chunk_results = {}  # index -> result
            claimed = set()  # indices claimed by a worker
            lock = threading.Lock()

            def claim_chunk(idx):
                """Atomically claim a chunk index. Returns True if claimed."""
                with lock:
                    if idx in claimed:
                        return False
                    claimed.add(idx)
                    return True

            def store_result(idx, result):
                """Store a chunk result."""
                with lock:
                    chunk_results[idx] = result

            def api_worker():
                """Process chunks from the front using API (primary + fallback models)."""
                for i in range(num_chunks):
                    # Stop if all chunks are claimed
                    with lock:
                        if len(claimed) >= num_chunks:
                            break
                    if not claim_chunk(i):
                        continue

                    chunk = chunks[i]
                    print(f"  [API] Transcribing chunk {i+1}/{num_chunks} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s)...")

                    result = None
                    try:
                        result = self._transcribe_single_file(chunk['file_path'])
                    except Exception as e:
                        if self._is_rate_limit_error(e):
                            # Try fallback models
                            fallback_result = self._retry_with_fallback_models(
                                chunk['file_path'],
                                f" for chunk {i+1}"
                            )
                            if fallback_result:
                                result = fallback_result
                            else:
                                # Release chunk for local worker to handle
                                with lock:
                                    claimed.discard(i)
                                print(f"  [API] All models rate limited on chunk {i+1}, releasing for local worker")
                                # Brief pause to avoid hammering rate-limited API
                                time.sleep(2)
                                continue
                        else:
                            print(f"  [API] Warning: Failed to transcribe chunk {i+1}: {e}")

                    if result and result.get('segments'):
                        store_result(i, result)

            def local_worker():
                """Process chunks from the back using local transcription."""
                # Small delay to let API worker claim initial chunks
                time.sleep(1)
                for i in range(num_chunks - 1, -1, -1):
                    # Stop if all chunks are claimed
                    with lock:
                        if len(claimed) >= num_chunks:
                            break
                    if not claim_chunk(i):
                        continue

                    chunk = chunks[i]
                    print(f"  [Local] Transcribing chunk {i+1}/{num_chunks} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s)...")

                    try:
                        result = self._transcribe_chunk_locally(chunk['file_path'])
                        if result and result.get('segments'):
                            store_result(i, result)
                    except Exception as e:
                        print(f"  [Local] Warning: Failed to transcribe chunk {i+1}: {e}")

            # Run API and local workers concurrently
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="transcribe") as executor:
                api_future = executor.submit(api_worker)
                local_future = executor.submit(local_worker)

                # Wait for both to complete
                api_future.result()
                local_future.result()

            # Check for any unclaimed or failed chunks and retry locally
            missing_chunks = [i for i in range(num_chunks) if i not in chunk_results]
            if missing_chunks:
                print(f"  Retrying {len(missing_chunks)} missing chunks locally...")
                for i in missing_chunks:
                    chunk = chunks[i]
                    try:
                        result = self._transcribe_chunk_locally(chunk['file_path'])
                        if result and result.get('segments'):
                            chunk_results[i] = result
                    except Exception as e:
                        print(f"  Warning: Final retry failed for chunk {i+1}: {e}")

            # Assemble results in order
            all_segments = []
            full_text_parts = []
            total_duration = 0
            language = 'en'

            for i in range(num_chunks):
                result = chunk_results.get(i)
                if result and result.get('segments'):
                    offset = chunks[i]['start_time']
                    for segment in result['segments']:
                        adjusted_segment = {
                            'start': segment['start'] + offset,
                            'end': segment['end'] + offset,
                            'text': segment['text'],
                            'speaker': segment.get('speaker'),
                            'confidence': segment.get('confidence', 1.0)
                        }
                        all_segments.append(adjusted_segment)

                    full_text_parts.append(result.get('text', ''))
                    language = result.get('language', language)

                total_duration = max(total_duration, chunks[i]['end_time'])

            # Merge overlapping segments (from overlap_seconds)
            all_segments = self._merge_overlapping_segments(all_segments)

            api_count = sum(1 for r in chunk_results.values()
                          if r and 'api' in r.get('provider', ''))
            local_count = num_chunks - api_count
            print(f"✓ Chunked transcription complete: {len(all_segments)} segments, "
                  f"{total_duration:.1f}s total (API: {api_count}, local: {local_count})")

            return {
                'segments': all_segments,
                'language': language,
                'text': ' '.join(full_text_parts),
                'provider': f'{self.api_provider}-api-chunked',
                'duration': total_duration
            }

        except Exception as e:
            print(f"Chunked transcription failed: {e}")
            print("Falling back to local Whisper")
            return self.transcribe_with_faster_whisper(audio_path) if self.use_faster_whisper else self.transcribe_with_whisper(audio_path)

        finally:
            # Clean up temp directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _merge_overlapping_segments(self, segments: List[Dict[str, Any]],
                                    overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Merge segments that overlap due to chunk boundaries.

        Args:
            segments: List of segments (may have overlaps)
            overlap_threshold: Time threshold to detect overlapping segments

        Returns:
            Deduplicated and merged segment list
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s['start'])

        merged = [sorted_segments[0]]

        for current in sorted_segments[1:]:
            prev = merged[-1]

            # Check if segments overlap significantly
            if current['start'] < prev['end'] + overlap_threshold:
                # If text is similar (likely duplicate from overlap), skip
                if self._texts_similar(prev['text'], current['text']):
                    # Extend end time if current goes further
                    if current['end'] > prev['end']:
                        prev['end'] = current['end']
                    continue

            merged.append(current)

        return merged

    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Check if two texts are similar (for deduplication).

        Uses simple word overlap ratio.

        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (0-1)

        Returns:
            True if texts are similar
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return (intersection / union) >= threshold if union > 0 else False

    def transcribe_episode(self, episode_id: int) -> bool:
        """Transcribe a single episode and store results."""
        episodes = self.db.get_episodes_by_status('downloaded')
        episode = next((ep for ep in episodes if ep['id'] == episode_id), None)
        
        if not episode:
            episode_row = self.db.conn.execute(
                "SELECT id, file_path, title, status FROM episodes WHERE id = ?",
                [episode_id]
            ).fetchone()
            if not episode_row:
                print(f"Episode {episode_id} not found")
                return False
            episode = {
                "id": episode_row[0],
                "file_path": episode_row[1],
                "title": episode_row[2],
                "status": episode_row[3],
            }

        if not episode['file_path'] or not Path(episode['file_path']).exists():
            print(f"Audio file not found: {episode.get('file_path')}")
            self.db.update_episode_status(episode_id, 'failed')
            return False

        print(f"Transcribing: {episode['title']}")

        # Choose transcription method (priority: API > Parakeet > faster-whisper > Whisper)
        if self.use_api:
            result = self.transcribe_with_api(episode['file_path'])
        elif self.use_parakeet:
            result = self.transcribe_with_parakeet(episode['file_path'])
        elif self.use_faster_whisper:
            result = self.transcribe_with_faster_whisper(episode['file_path'])
        else:
            result = self.transcribe_with_whisper(episode['file_path'])
        
        if not result:
            return False

        # Check if we have a reasonable transcript (at least some content)
        segments = result.get('segments', [])
        if not segments:
            print(f"⚠ Transcription produced no segments for: {episode['title']}")
            return False

        # Store transcript segments in database
        self.db.add_transcript_segments(episode_id, segments)

        # Update episode status
        self.db.update_episode_status(episode_id, 'transcribed')

        print(f"✓ Transcribed: {episode['title']}")
        return True

    def transcribe_all_pending(self) -> int:
        """Transcribe all episodes with 'downloaded' status."""
        episodes = self.db.get_episodes_by_status('downloaded')
        missing_transcripts = []
        for episode in self.db.get_episodes_by_status('transcribed'):
            if not self.db.get_transcripts_for_episode(episode['id']):
                missing_transcripts.append(episode)
        retry_failed = []
        for episode in self.db.get_episodes_by_status('failed'):
            file_path = episode.get('file_path')
            if file_path and Path(file_path).exists():
                retry_failed.append(episode)
        episodes.extend(missing_transcripts)
        episodes.extend(retry_failed)
        seen = set()
        unique_episodes = []
        for episode in episodes:
            if episode['id'] not in seen:
                seen.add(episode['id'])
                unique_episodes.append(episode)
        transcribed_count = 0
        
        for episode in unique_episodes:
            if self.transcribe_episode(episode['id']):
                transcribed_count += 1
            
        return transcribed_count

    def _retry_with_fallback_models(
        self,
        audio_path: str,
        context_label: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Retry transcription with fallback models when rate limited."""
        fallback_models = self._get_api_fallback_models()
        if not fallback_models:
            return None

        original_model = self.api_model
        try:
            for fallback_model in fallback_models:
                self.api_model = fallback_model
                try:
                    result = self._transcribe_single_file(audio_path)
                    if result:
                        return result
                except Exception as fallback_error:
                    print(
                        f"  ⚠ Fallback model '{fallback_model}' also rate limited"
                        f"{context_label}"
                    )
        finally:
            self.api_model = original_model

        return None

    def _get_api_fallback_models(self) -> List[str]:
        """Get unique fallback models for API transcription."""
        fallback_models = []
        for model in self.api_fallback_models:
            if model and model != self.api_model and model not in fallback_models:
                fallback_models.append(model)
        return fallback_models

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Check whether an error likely represents a rate limit response."""
        error_str = str(error)
        return "429" in error_str or "rate_limit" in error_str or "413" in error_str

    def get_full_transcript(self, episode_id: int) -> str:
        """Get the full transcript text for an episode."""
        segments = self.db.get_transcripts_for_episode(episode_id)
        return "\n".join(segment['text'] for segment in segments)

    def export_transcript(self, episode_id: int, format: str = "txt") -> str:
        """Export transcript in various formats."""
        segments = self.db.get_transcripts_for_episode(episode_id)
        
        if format == "txt":
            return "\n".join(segment['text'] for segment in segments)
        
        elif format == "srt":
            srt_content = []
            for i, segment in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(segment['timestamp_start'] or 0)
                end_time = self._seconds_to_srt_time(segment['timestamp_end'] or 0)
                srt_content.append(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n")
            return "\n".join(srt_content)
        
        elif format == "json":
            return json.dumps(segments, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

