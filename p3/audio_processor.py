"""Audio processing utilities for P³ using ffmpeg."""

import subprocess
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio cutting, stitching, and validation using ffmpeg."""

    def __init__(self):
        """Initialize audio processor."""
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify ffmpeg and ffprobe are available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
            subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "ffmpeg and ffprobe are required but not found. "
                "Please install ffmpeg: https://ffmpeg.org/download.html"
            ) from e

    def get_audio_duration(self, file_path: str) -> float:
        """Get duration of audio file in seconds using ffprobe.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json",
                    file_path
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                check=True
            )
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except Exception as e:
            logger.error(f"Failed to get duration for {file_path}: {e}")
            raise

    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed audio file information.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with codec, sample_rate, channels, bitrate, duration
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate",
                    "-show_entries", "format=duration,size",
                    "-of", "json",
                    file_path
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                check=True
            )
            data = json.loads(result.stdout)

            stream = data.get("streams", [{}])[0]
            format_info = data.get("format", {})

            return {
                "codec": stream.get("codec_name"),
                "sample_rate": int(stream.get("sample_rate", 0)),
                "channels": int(stream.get("channels", 0)),
                "bitrate": int(stream.get("bit_rate", 0)) if stream.get("bit_rate") else None,
                "duration": float(format_info.get("duration", 0)),
                "size_bytes": int(format_info.get("size", 0))
            }
        except Exception as e:
            logger.error(f"Failed to get info for {file_path}: {e}")
            raise

    def cut_audio_segment(self, input_path: str, output_path: str,
                         start_time: float, end_time: float) -> str:
        """Cut a segment from audio file with frame-accurate precision.

        Uses re-encoding for precise cuts (not -c copy which only cuts at keyframes).

        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Path to output file
        """
        duration = end_time - start_time

        if duration <= 0:
            raise ValueError(f"Invalid duration: {duration}s (start={start_time}, end={end_time})")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", input_path,
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-c:a", "libmp3lame",
                    "-q:a", "2",  # High quality VBR
                    "-y",  # Overwrite output file
                    output_path
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                check=True
            )
            logger.info(f"Cut segment: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s) -> {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cut audio segment: {e.stderr}")
            raise

    def stitch_audio_files(self, input_files: List[str], output_path: str) -> str:
        """Stitch multiple audio files together.

        Re-encodes with consistent format to avoid sample rate mismatches.
        Resamples to 44.1kHz and converts to mono for consistency.

        Args:
            input_files: List of paths to input audio files
            output_path: Path to output audio file

        Returns:
            Path to output file
        """
        if not input_files:
            raise ValueError("No input files provided")

        if len(input_files) == 1:
            # Just copy the file if only one input
            shutil.copy2(input_files[0], output_path)
            return output_path

        # Create concat file list
        concat_file = Path(output_path).parent / "concat_list.txt"

        try:
            with open(concat_file, 'w') as f:
                for file_path in input_files:
                    # Use absolute paths and escape single quotes
                    abs_path = os.path.abspath(file_path)
                    f.write(f"file '{abs_path}'\n")

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Stitch files with re-encoding for consistency
            subprocess.run(
                [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_file),
                    "-ar", "44100",  # Resample to 44.1kHz
                    "-ac", "1",  # Mono output
                    "-c:a", "libmp3lame",
                    "-q:a", "2",  # High quality VBR
                    "-y",
                    output_path
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                check=True
            )
            logger.info(f"Stitched {len(input_files)} files -> {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stitch audio files: {e.stderr}")
            raise
        finally:
            # Clean up concat file
            if concat_file.exists():
                concat_file.unlink()

    def validate_audio_output(self, file_path: str,
                             expected_duration_min: float = None,
                             expected_duration_max: float = None) -> Dict[str, Any]:
        """Comprehensive audio file validation.

        Args:
            file_path: Path to audio file to validate
            expected_duration_min: Minimum expected duration in seconds
            expected_duration_max: Maximum expected duration in seconds

        Returns:
            Dictionary with validation results:
            - valid: bool
            - issues: List[str] (critical errors)
            - warnings: List[str] (non-critical issues)
            - file_path: str
            - size_mb: float
            - duration: float
            - codec: str
            - sample_rate: int
            - channels: int
        """
        issues = []
        warnings = []

        # Check file existence
        if not os.path.exists(file_path):
            return {
                "valid": False,
                "issues": [f"File does not exist: {file_path}"],
                "warnings": [],
                "file_path": file_path
            }

        # Check file size
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)

        if size_mb < 0.1:
            issues.append(f"File too small: {size_mb:.2f}MB")
        elif size_mb > 100:
            warnings.append(f"File very large: {size_mb:.2f}MB")

        # Get audio info
        try:
            info = self.get_audio_info(file_path)
        except Exception as e:
            issues.append(f"Failed to read audio info: {e}")
            return {
                "valid": False,
                "issues": issues,
                "warnings": warnings,
                "file_path": file_path,
                "size_mb": size_mb
            }

        duration = info["duration"]
        codec = info["codec"]
        sample_rate = info["sample_rate"]
        channels = info["channels"]

        # Validate duration
        if expected_duration_min and duration < expected_duration_min:
            issues.append(
                f"Duration too short: {duration:.1f}s (expected >= {expected_duration_min:.1f}s)"
            )
        if expected_duration_max and duration > expected_duration_max:
            warnings.append(
                f"Duration too long: {duration:.1f}s (expected <= {expected_duration_max:.1f}s)"
            )

        # Validate codec
        expected_codecs = ["mp3", "aac", "opus"]
        if codec not in expected_codecs:
            warnings.append(f"Unexpected codec: {codec} (expected one of {expected_codecs})")

        # Validate sample rate
        common_rates = [44100, 48000, 22050, 24000]
        if sample_rate not in common_rates:
            warnings.append(f"Unusual sample rate: {sample_rate}Hz")

        # Validate channels
        if channels not in [1, 2]:
            warnings.append(f"Unusual channel count: {channels}")

        # Check for excessive silence (if file is otherwise valid)
        if not issues:
            silence_count = self._detect_silence_periods(file_path)
            if silence_count > 20:
                warnings.append(f"Many silence periods detected: {silence_count}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "file_path": file_path,
            "size_mb": size_mb,
            "duration": duration,
            "codec": codec,
            "sample_rate": sample_rate,
            "channels": channels
        }

    def _detect_silence_periods(self, file_path: str,
                                silence_threshold_db: int = -40,
                                min_silence_duration: float = 2.0) -> int:
        """Detect number of silence periods in audio file.

        Args:
            file_path: Path to audio file
            silence_threshold_db: Silence threshold in dB
            min_silence_duration: Minimum silence duration in seconds

        Returns:
            Number of silence periods detected
        """
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i", file_path,
                    "-af", f"silencedetect=noise={silence_threshold_db}dB:d={min_silence_duration}",
                    "-f", "null",
                    "-"
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            # Count "silence_end" occurrences in stderr
            silence_count = result.stderr.count("silence_end")
            return silence_count
        except Exception as e:
            logger.warning(f"Failed to detect silence: {e}")
            return 0

    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in megabytes.

        Args:
            file_path: Path to file

        Returns:
            File size in MB
        """
        return os.path.getsize(file_path) / (1024 * 1024)

    def estimate_chunk_duration_for_size(self, file_path: str,
                                         target_size_mb: float = 20.0) -> float:
        """Estimate chunk duration to achieve target file size.

        Args:
            file_path: Path to audio file
            target_size_mb: Target chunk size in MB (default 20MB for safety margin)

        Returns:
            Estimated chunk duration in seconds
        """
        info = self.get_audio_info(file_path)
        file_size_mb = info["size_bytes"] / (1024 * 1024)
        duration = info["duration"]

        if file_size_mb <= 0 or duration <= 0:
            return 600.0  # Default to 10 minutes if we can't calculate

        # Calculate MB per second
        mb_per_second = file_size_mb / duration

        # Calculate duration for target size
        chunk_duration = target_size_mb / mb_per_second

        # Ensure minimum chunk size of 60 seconds
        return max(60.0, chunk_duration)

    def split_audio_into_chunks(self, input_path: str, output_dir: str,
                                max_size_mb: float = 20.0,
                                overlap_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """Split audio file into chunks under the size limit.

        Uses time-based splitting with optional overlap for better transcription
        continuity at chunk boundaries.

        Args:
            input_path: Path to input audio file
            output_dir: Directory to write chunks
            max_size_mb: Maximum chunk size in MB (default 20MB for safety)
            overlap_seconds: Overlap between chunks in seconds

        Returns:
            List of chunk info dictionaries with:
            - file_path: Path to chunk file
            - start_time: Original start time in seconds
            - end_time: Original end time in seconds
            - duration: Chunk duration in seconds
            - chunk_index: 0-based index
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get file info
        info = self.get_audio_info(input_path)
        total_duration = info["duration"]
        file_size_mb = info["size_bytes"] / (1024 * 1024)

        # If file is already under limit, just copy it
        if file_size_mb <= max_size_mb:
            output_path = str(Path(output_dir) / "chunk_000.mp3")
            shutil.copy2(input_path, output_path)
            return [{
                "file_path": output_path,
                "start_time": 0.0,
                "end_time": total_duration,
                "duration": total_duration,
                "chunk_index": 0
            }]

        # Calculate chunk duration
        chunk_duration = self.estimate_chunk_duration_for_size(input_path, max_size_mb)

        chunks = []
        current_start = 0.0
        chunk_index = 0

        # Calculate expected number of chunks for progress display
        expected_chunks = int((total_duration / chunk_duration) + 1)
        print(f"  Splitting {file_size_mb:.1f}MB file ({total_duration/60:.1f} min) into ~{expected_chunks} chunks...", flush=True)

        while current_start < total_duration:
            # Calculate end time for this chunk
            current_end = min(current_start + chunk_duration, total_duration)
            actual_duration = current_end - current_start

            # Generate output path
            output_path = str(Path(output_dir) / f"chunk_{chunk_index:03d}.mp3")

            # Cut the chunk - show progress
            print(f"    Creating chunk {chunk_index + 1}/{expected_chunks}: {current_start/60:.1f}m - {current_end/60:.1f}m...", end='', flush=True)

            try:
                self.cut_audio_segment(input_path, output_path, current_start, current_end)

                # Verify the chunk was created
                if os.path.exists(output_path):
                    chunk_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f" ({chunk_size_mb:.1f}MB)", flush=True)
                else:
                    print(" [FAILED - no output]", flush=True)
                    raise Exception(f"Chunk file was not created: {output_path}")

                chunks.append({
                    "file_path": output_path,
                    "start_time": current_start,
                    "end_time": current_end,
                    "duration": actual_duration,
                    "chunk_index": chunk_index
                })

                logger.info(f"Created chunk {chunk_index}: {current_start:.1f}s - {current_end:.1f}s ({actual_duration:.1f}s)")

            except Exception as e:
                logger.error(f"Failed to create chunk {chunk_index}: {e}")
                raise

            # Move to next chunk (with overlap for continuity)
            # Break if we've processed to the end to avoid infinite loop
            if current_end >= total_duration:
                break
            current_start = current_end - overlap_seconds
            chunk_index += 1

        print(f"  ✓ Created {len(chunks)} chunks successfully", flush=True)
        logger.info(f"Split {input_path} into {len(chunks)} chunks (total: {total_duration:.1f}s)")
        return chunks

    def extract_timestamp_range(self, input_path: str, output_path: str,
                                start_sec: float, end_sec: float) -> str:
        """Extract a specific timestamp range from audio file.

        This is a convenience wrapper around cut_audio_segment for extracting
        clips based on transcript timestamps.

        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            start_sec: Start time in seconds
            end_sec: End time in seconds

        Returns:
            Path to output file
        """
        return self.cut_audio_segment(input_path, output_path, start_sec, end_sec)

    def needs_chunking(self, file_path: str, max_size_mb: float = 25.0) -> bool:
        """Check if file needs chunking for API upload.

        Args:
            file_path: Path to audio file
            max_size_mb: Maximum allowed size in MB

        Returns:
            True if file exceeds size limit
        """
        return self.get_file_size_mb(file_path) > max_size_mb

