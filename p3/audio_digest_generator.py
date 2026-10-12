"""Generate audio digests by selecting and stitching podcast clips."""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os

from p3.database import P3Database
from p3.clip_generator import ClipGenerator
from p3.clip_selector import ClipSelector
from p3.audio_processor import AudioProcessor
from p3.tts_generator import TTSGenerator
from p3.narration_generator import NarrationGenerator
from p3.windows_utils import create_windows_temp_dir
from p3.storage_monitor import StorageMonitor, check_storage_before_operation
from p3.progress_helper import P3Progress
from p3.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class AudioDigestGenerator:
    """Orchestrates audio digest generation from podcast episodes."""

    def __init__(self, db: P3Database, model: str = "llama3.2:3b",
                 min_clip_duration: float = 30.0, max_clip_duration: float = 180.0,
                 max_candidates_per_episode: int = 40, max_clips_per_episode: int = 20,
                 enable_narration: bool = True, tts_voice: str = "en_US-lessac-medium",
                 use_tts_api: bool = False, tts_api_provider: str = None,
                 tts_api_model: str = None, tts_api_key_file: str = None,
                 llm_provider: str = "ollama", llm_api_key_file: str = None,
                 piper_voice: str = "en_US-lessac-medium",
                 ollama_base_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3.2:3b",
                 model_fallbacks: Optional[List[str]] = None):
        """Initialize audio digest generator.

        Args:
            db: Database instance
            model: LLM model to use for clip selection (API model or local Ollama model)
            min_clip_duration: Minimum clip duration in seconds
            max_clip_duration: Maximum clip duration in seconds
            max_candidates_per_episode: Max clip candidates to generate per episode for LLM selection
            max_clips_per_episode: Maximum clips from one episode in final output (diversity)
            enable_narration: Whether to add TTS narration between clips
            tts_voice: TTS voice model to use (for local Piper or API provider-specific)
            use_tts_api: Whether to use external API for TTS
            tts_api_provider: TTS API provider ("groq", "openai")
            tts_api_model: TTS model to use for API
            tts_api_key_file: Path to TTS API key file
            llm_provider: LLM provider ("ollama", "groq", "openai", etc.)
            llm_api_key_file: Path to LLM API key file (for cloud providers)
            piper_voice: Local Piper voice for TTS fallback
            ollama_base_url: Base URL for local Ollama (used for direct or fallback calls)
            ollama_model: Local Ollama model for LLM fallback (from feeds.yaml llm_model)
            model_fallbacks: Ordered list of API model fallbacks to try on rate limits
        """
        self.db = db
        self.max_candidates_per_episode = max_candidates_per_episode
        self.max_clips_per_episode = max_clips_per_episode
        self.enable_narration = enable_narration
        self.clip_generator = ClipGenerator(min_duration=min_clip_duration, max_duration=max_clip_duration)
        self.clip_selector = ClipSelector(
            model=model,
            provider=llm_provider,
            api_key_file=llm_api_key_file,
            base_url=ollama_base_url,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            model_fallbacks=model_fallbacks
        )
        self.audio_processor = AudioProcessor()

        # TTS components (only initialize if enabled)
        self.tts_generator = None
        self.narration_generator = None
        if enable_narration:
            self.tts_generator = TTSGenerator(
                voice=tts_voice,
                use_api=use_tts_api,
                api_provider=tts_api_provider,
                api_model=tts_api_model,
                api_key_file=tts_api_key_file,
                piper_voice=piper_voice
            )
            if self.tts_generator.is_available():
                self.narration_generator = NarrationGenerator(
                    model=model,
                    provider=llm_provider,
                    api_key_file=llm_api_key_file,
                    base_url=ollama_base_url,
                    ollama_base_url=ollama_base_url,
                    ollama_model=ollama_model,
                    model_fallbacks=model_fallbacks
                )
                logger.info(f"TTS narration enabled ({'API' if use_tts_api else 'local'})")
            else:
                logger.warning("TTS not available - narration will be disabled")
                self.enable_narration = False

    def generate_digest_for_date(self, date: datetime, target_duration: int = None,
                                 output_dir: str = "outputs/audio_digests") -> Optional[str]:
        """Generate audio digest for all episodes from a specific date.

        Args:
            date: Date to generate digest for
            target_duration: Target duration in seconds (default: 1/10th of total input)
            output_dir: Directory to save output file

        Returns:
            Path to generated audio file, or None if no episodes found
        """
        logger.info(f"Generating audio digest for {date.date()}")

        # Check storage before starting
        try:
            check_storage_before_operation(required_gb=1.0)
        except Exception as e:
            logger.error(f"Storage check failed: {e}")
            print(f"\n[red]✗ {e}[/red]")
            print(f"[yellow]Free up disk space or run: p3 cleanup[/yellow]\n")
            return None

        # Initialize storage and performance monitoring
        storage_monitor = StorageMonitor(min_free_gb=0.5, warn_threshold_gb=1.0)
        storage_monitor.start_monitoring()
        perf_monitor = PerformanceMonitor(sample_interval=10.0)
        perf_monitor.start()
        print(f"      Storage: {storage_monitor.initial_free_gb:.1f} GB free")

        print(f"[1/6] Finding transcribed episodes for {date.date()}...")

        # Get all transcribed episodes for this date
        episodes = self._get_transcribed_episodes_for_date(date)

        if not episodes:
            logger.warning(f"No transcribed episodes found for {date.date()}")
            return None

        logger.info(f"Found {len(episodes)} episodes for {date.date()}")
        print(f"      Found {len(episodes)} episodes")

        # Warn if summaries exist without corresponding processed episodes (audio may miss them)
        try:
            summaries = self.db.get_summaries_by_date(date)
            summary_episode_ids = {summary["episode_id"] for summary in summaries}
            processed_episode_ids = {episode["id"] for episode in episodes}
            missing_in_audio = summary_episode_ids - processed_episode_ids
            if missing_in_audio:
                logger.warning(
                    "Some summaries have no processed episode available for audio digest: "
                    f"{sorted(missing_in_audio)}"
                )
                print(
                    f"      ⚠ Summaries missing from audio digest: {sorted(missing_in_audio)}"
                )
        except Exception as e:
            logger.warning(f"Could not compare summaries to audio digest episodes: {e}")

        # Calculate target duration if not specified (1/10th of total)
        if target_duration is None:
            # Use 'or 0' to handle None values (get() default only applies when key is missing)
            total_duration = sum((ep.get("duration_seconds") or 0) for ep in episodes)
            if total_duration == 0:
                # Estimate from transcripts if duration not stored
                total_duration = sum(self._estimate_episode_duration(ep["id"]) for ep in episodes)
            target_duration = max(60, int(total_duration / 10))  # At least 60 seconds
            logger.info(f"Auto-calculated target duration: {target_duration}s (1/10th of {total_duration}s)")
            total_mins = total_duration // 60
            target_mins = target_duration // 60
            print(f"      Input: {total_mins} min → Target: {target_mins} min (1/10th reduction)")

        # Generate candidates from all episodes
        print(f"[2/6] Generating clip candidates...")
        all_candidates = []
        episode_info = []

        for episode in episodes:
            transcripts = self.db.get_transcripts_for_episode(episode["id"])
            if not transcripts:
                logger.warning(f"No transcripts for episode {episode['id']}")
                continue

            # Generate candidates for this episode
            candidates = self.clip_generator.generate_candidates(
                transcripts,
                max_candidates=self.max_candidates_per_episode
            )

            # Add episode_id to each candidate
            for candidate in candidates:
                candidate["episode_id"] = episode["id"]

            all_candidates.extend(candidates)

            # Store episode info for LLM context
            episode_info.append({
                "id": episode["id"],
                "title": episode["title"],
                "podcast_title": episode["podcast_title"],
                "file_path": episode["file_path"]
            })

        if not all_candidates:
            logger.warning("No candidate clips generated")
            return None

        logger.info(f"Generated {len(all_candidates)} total candidate clips")
        print(f"      Generated {len(all_candidates)} candidate clips")

        # Calculate max_clips based on target duration and average/min clip length
        # Ensure enough clips to hit target duration even when clips are near minimum length
        avg_clip_duration = (self.clip_generator.min_duration + self.clip_generator.max_duration) / 2
        buffer_clips = int(target_duration / avg_clip_duration * 1.5)
        min_required_clips = int((target_duration + self.clip_generator.min_duration - 1)
                                 / self.clip_generator.min_duration)
        effective_max_clips_per_episode = self.max_clips_per_episode
        max_possible_clips = effective_max_clips_per_episode * max(1, len(episodes))
        max_clips = max(3, buffer_clips, min_required_clips)
        if max_possible_clips < max_clips:
            boosted_per_episode = math.ceil(min_required_clips / max(1, len(episodes)))
            if boosted_per_episode > effective_max_clips_per_episode:
                effective_max_clips_per_episode = boosted_per_episode
                max_possible_clips = effective_max_clips_per_episode * max(1, len(episodes))
            logger.warning(
                "Clip limits may be too low to reach target duration: "
                f"max_possible={max_possible_clips}, needed={max_clips}"
            )
            max_clips = max_possible_clips

        logger.info(f"Target duration: {target_duration}s, calculated max_clips: {max_clips}")

        # Select best clips using LLM
        print(f"[3/6] Selecting best {max_clips} clips using LLM...")
        selected_clips = self.clip_selector.select_clips(
            all_candidates,
            episode_info,
            target_duration=target_duration,
            max_clips=max_clips,
            max_clips_per_episode=effective_max_clips_per_episode,
            min_clip_duration=self.clip_generator.min_duration,
            max_clip_duration=self.clip_generator.max_duration
        )

        if not selected_clips:
            logger.warning("No clips selected")
            return None

        # Validate selection
        is_valid, error = self.clip_generator.validate_clip_selection(
            selected_clips,
            max_clips_per_episode=effective_max_clips_per_episode
        )
        if not is_valid:
            logger.error(f"Clip selection validation failed: {error}")
            # Continue anyway with a warning

        logger.info(f"Selected {len(selected_clips)} clips")
        print(f"      Selected {len(selected_clips)} clips")

        # Cut and save individual clips
        print(f"[4/6] Extracting {len(selected_clips)} clips from source audio...")
        clip_files = []
        successful_clips = []
        narration_files = []  # Initialize before try block to avoid NameError in finally
        temp_dir = create_windows_temp_dir(prefix="p3_clips_")

        try:
            # Create progress bar for clip extraction
            progress = P3Progress.create_progress_bar("Extracting clips")
            with progress:
                task = progress.add_task("Extracting clips", total=len(selected_clips))

                for i, clip in enumerate(selected_clips):
                    # Check storage periodically
                    is_ok, warning = storage_monitor.check_during_generation()
                    if warning:
                        logger.warning(warning)
                    if not is_ok:
                        logger.error(warning)
                        print(f"\n      [red]✗ {warning}[/red]")
                        break

                    # Find episode file path
                    clip_episode_id = clip.get("episode_id")
                    if clip_episode_id is None:
                        logger.error(f"Clip {i} missing episode_id, skipping")
                        progress.update(task, advance=1)
                        continue

                    episode = next((ep for ep in episode_info if ep["id"] == clip_episode_id), None)
                    if episode is None:
                        logger.error(f"Episode {clip_episode_id} not found in episode_info, skipping clip {i}")
                        progress.update(task, advance=1)
                        continue

                    episode_file = episode["file_path"]

                    if not episode_file or not os.path.exists(episode_file):
                        logger.error(f"Episode file not found: {episode_file}")
                        progress.update(task, advance=1)
                        continue

                    # Cut clip
                    clip_file = temp_dir / f"clip_{i:03d}.mp3"
                    self.audio_processor.cut_audio_segment(
                        episode_file,
                        str(clip_file),
                        clip["start_sec"],
                        clip["end_sec"]
                    )

                    clip_files.append(str(clip_file))
                    successful_clips.append(clip)

                    # Save clip to database (file_path is None since these are temporary files)
                    self.db.add_audio_clip(
                        episode_id=clip["episode_id"],
                        start_sec=clip["start_sec"],
                        end_sec=clip["end_sec"],
                        text=clip.get("text"),
                        topic=clip.get("topic"),
                        score=clip.get("score"),
                        file_path=None
                    )

                    # Update progress
                    progress.update(task, advance=1)

            if not clip_files:
                logger.error("No clip files generated")
                return None

            print(f"      Extracted {len(clip_files)} clips                    ")  # Clear progress line

            # Generate narration if enabled
            if self.enable_narration and self.narration_generator and self.tts_generator:
                print(f"[5/6] Generating TTS narration for smooth transitions...")
                logger.info("Generating narration scripts...")
                narration = self.narration_generator.generate_narration_scripts(
                    successful_clips, episode_info
                )

                # Validate transition count
                transitions = narration.get("transitions", [])
                expected_transitions = max(0, len(successful_clips) - 1)
                if len(transitions) != expected_transitions:
                    logger.warning(
                        f"Expected {expected_transitions} transitions but got {len(transitions)}. "
                        f"Some transitions may be missing or extra transitions will be ignored."
                    )

                # Generate TTS audio for intro
                if narration.get("intro"):
                    print(f"      Generating intro narration...")
                    intro_file = temp_dir / "narration_intro.mp3"
                    intro_path = self.tts_generator.generate_speech(
                        narration["intro"], str(intro_file), label="intro"
                    )
                    if intro_path:
                        narration_files.append(("intro", intro_path))

                # Generate TTS audio for transitions
                transitions = narration.get("transitions", [])
                for i, transition_text in enumerate(transitions):
                    if transition_text:
                        print(f"      Generating transition {i+1}/{len(transitions)}...", end='\r')
                        transition_file = temp_dir / f"narration_transition_{i:03d}.mp3"
                        transition_path = self.tts_generator.generate_speech(
                            transition_text, str(transition_file), label=f"transition {i+1}/{len(transitions)}"
                        )
                        if transition_path:
                            narration_files.append((f"transition_{i}", transition_path))

                logger.info(f"Generated {len(narration_files)} narration segments")
                print(f"      Generated {len(narration_files)} narration segments      ")
            else:
                print(f"[5/6] Skipping narration (disabled)")

            # Stitch clips together with narration
            output_path = self._get_output_path(date, output_dir)

            if narration_files:
                print(f"[6/6] Stitching {len(clip_files)} clips + {len(narration_files)} narrations...")
                # Build interleaved file list: intro, clip1, transition1, clip2, transition2, ...
                interleaved_files = []

                # Add intro
                intro = next((path for label, path in narration_files if label == "intro"), None)
                if intro:
                    interleaved_files.append(intro)

                # Interleave clips with transitions
                for i, clip_file in enumerate(clip_files):
                    interleaved_files.append(clip_file)

                    # Add transition after clip (if not last clip)
                    if i < len(clip_files) - 1:
                        transition = next(
                            (path for label, path in narration_files if label == f"transition_{i}"),
                            None
                        )
                        if transition:
                            interleaved_files.append(transition)

                logger.info(f"Stitching {len(interleaved_files)} segments (clips + narration)...")
                final_file = self.audio_processor.stitch_audio_files(interleaved_files, output_path)
            else:
                # No narration - stitch clips only
                print(f"[6/6] Stitching {len(clip_files)} clips...")
                final_file = self.audio_processor.stitch_audio_files(clip_files, output_path)

            print(f"      Validating output...")
            # Validate output
            validation = self.audio_processor.validate_audio_output(
                final_file,
                expected_duration_min=target_duration * 0.7,
                expected_duration_max=target_duration * 1.3
            )

            if not validation["valid"]:
                logger.error(f"Audio validation failed: {validation['issues']}")
            elif validation["warnings"]:
                logger.warning(f"Audio validation warnings: {validation['warnings']}")

            # Save digest metadata to database
            self.db.add_audio_digest(
                digest_date=date,
                file_path=final_file,
                duration_sec=validation.get("duration"),
                target_duration_sec=target_duration,
                clip_count=len(selected_clips),
                episode_count=len(episodes)
            )

            logger.info(f"Audio digest saved: {final_file}")
            logger.info(f"Duration: {validation.get('duration', 0):.1f}s (target: {target_duration}s)")
            logger.info(f"Clips: {len(selected_clips)} from {len(episodes)} episodes")

            # Show storage monitoring summary
            storage_summary = storage_monitor.format_summary()
            logger.info(storage_summary)
            print(f"      {storage_summary}")

            # Show performance monitoring summary
            perf_summary = perf_monitor.format_summary()
            logger.info(perf_summary)
            print(f"      {perf_summary}")

            return final_file

        finally:
            # Clean up temporary files with size tracking
            total_cleanup_size_mb = 0
            cleanup_count = 0

            # Clean up temporary clip files
            for clip_file in clip_files:
                try:
                    if os.path.exists(clip_file):
                        size_mb = os.path.getsize(clip_file) / (1024 * 1024)
                        total_cleanup_size_mb += size_mb
                        os.remove(clip_file)
                        cleanup_count += 1
                except Exception as e:
                    logger.debug(f"Failed to clean up temp file {clip_file}: {e}")

            # Clean up temporary narration files
            for _, narration_file in narration_files:
                try:
                    if os.path.exists(narration_file):
                        size_mb = os.path.getsize(narration_file) / (1024 * 1024)
                        total_cleanup_size_mb += size_mb
                        os.remove(narration_file)
                        cleanup_count += 1
                except Exception as e:
                    logger.debug(f"Failed to clean up temp file {narration_file}: {e}")

            # Remove temp directory
            try:
                temp_dir.rmdir()
            except Exception:
                pass  # Directory might not be empty, that's ok

            # Report cleanup
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} temp files ({total_cleanup_size_mb:.1f} MB)")
                print(f"      Cleaned up {cleanup_count} temp files ({total_cleanup_size_mb:.1f} MB)")

    def generate_digest_for_episodes(self, episode_ids: List[int],
                                    target_duration: int = None,
                                    output_path: str = None) -> Optional[str]:
        """Generate audio digest for specific episodes.

        Args:
            episode_ids: List of episode IDs to include
            target_duration: Target duration in seconds (default: 1/10th of total input)
            output_path: Full path to output file (default: auto-generated)

        Returns:
            Path to generated audio file, or None if failed
        """
        logger.info(f"Generating audio digest for {len(episode_ids)} episodes")

        # Get episode details
        episodes = []
        for ep_id in episode_ids:
            # We need to fetch episode details - let's add a helper method
            ep_data = self._get_episode_by_id(ep_id)
            if ep_data:
                episodes.append(ep_data)

        if not episodes:
            logger.error("No valid episodes found")
            return None

        # Use the main generation logic (similar to generate_digest_for_date)
        # For now, use the first episode's date as the digest date
        digest_date = episodes[0].get("date", datetime.now())

        # Calculate target duration if not specified
        if target_duration is None:
            # Use 'or 0' to handle None values (get() default only applies when key is missing)
            total_duration = sum((ep.get("duration_seconds") or 0) for ep in episodes)
            if total_duration == 0:
                total_duration = sum(self._estimate_episode_duration(ep["id"]) for ep in episodes)
            target_duration = max(60, int(total_duration / 10))

        # Generate candidates from specified episodes
        all_candidates = []
        episode_info = []

        for episode in episodes:
            transcripts = self.db.get_transcripts_for_episode(episode["id"])
            if not transcripts:
                continue

            candidates = self.clip_generator.generate_candidates(
                transcripts,
                max_candidates=self.max_candidates_per_episode
            )

            for candidate in candidates:
                candidate["episode_id"] = episode["id"]

            all_candidates.extend(candidates)
            episode_info.append({
                "id": episode["id"],
                "title": episode["title"],
                "podcast_title": episode.get("podcast_title", "Unknown"),
                "file_path": episode["file_path"]
            })

        if not all_candidates:
            logger.error("No candidate clips generated")
            return None

        # Calculate max_clips based on target duration and average/min clip length
        avg_clip_duration = (self.clip_generator.min_duration + self.clip_generator.max_duration) / 2
        buffer_clips = int(target_duration / avg_clip_duration * 1.5)
        min_required_clips = int((target_duration + self.clip_generator.min_duration - 1)
                                 / self.clip_generator.min_duration)
        effective_max_clips_per_episode = self.max_clips_per_episode
        max_possible_clips = effective_max_clips_per_episode * max(1, len(episodes))
        max_clips = max(3, buffer_clips, min_required_clips)
        if max_possible_clips < max_clips:
            boosted_per_episode = math.ceil(min_required_clips / max(1, len(episodes)))
            if boosted_per_episode > effective_max_clips_per_episode:
                effective_max_clips_per_episode = boosted_per_episode
                max_possible_clips = effective_max_clips_per_episode * max(1, len(episodes))
            logger.warning(
                "Clip limits may be too low to reach target duration: "
                f"max_possible={max_possible_clips}, needed={max_clips}"
            )
            max_clips = max_possible_clips

        logger.info(f"Target duration: {target_duration}s, calculated max_clips: {max_clips}")

        # Select clips using LLM
        selected_clips = self.clip_selector.select_clips(
            all_candidates,
            episode_info,
            target_duration=target_duration,
            max_clips=max_clips,
            max_clips_per_episode=effective_max_clips_per_episode,
            min_clip_duration=self.clip_generator.min_duration,
            max_clip_duration=self.clip_generator.max_duration
        )

        if not selected_clips:
            logger.error("No clips selected")
            return None

        # Cut clips and stitch
        clip_files = []
        successful_clips = []
        narration_files = []  # Initialize before try block to avoid NameError in finally
        temp_dir = create_windows_temp_dir(prefix="p3_clips_")

        try:
            for i, clip in enumerate(selected_clips):
                clip_episode_id = clip.get("episode_id")
                if clip_episode_id is None:
                    logger.error(f"Clip {i} missing episode_id, skipping")
                    continue

                episode = next((ep for ep in episode_info if ep["id"] == clip_episode_id), None)
                if episode is None:
                    logger.error(f"Episode {clip_episode_id} not found in episode_info, skipping clip {i}")
                    continue

                episode_file = episode["file_path"]

                if not episode_file or not os.path.exists(episode_file):
                    logger.error(f"Episode file not found: {episode_file}")
                    continue

                clip_file = temp_dir / f"clip_{i:03d}.mp3"
                self.audio_processor.cut_audio_segment(
                    episode_file, str(clip_file), clip["start_sec"], clip["end_sec"]
                )

                clip_files.append(str(clip_file))
                successful_clips.append(clip)

                # Save clip to database (file_path is None since these are temporary files)
                self.db.add_audio_clip(
                    episode_id=clip["episode_id"],
                    start_sec=clip["start_sec"],
                    end_sec=clip["end_sec"],
                    text=clip.get("text"),
                    topic=clip.get("topic"),
                    score=clip.get("score"),
                    file_path=None
                )

            if not clip_files:
                logger.error("No clip files generated")
                return None

            # Generate narration if enabled
            if self.enable_narration and self.narration_generator and self.tts_generator:
                logger.info("Generating narration scripts...")
                narration = self.narration_generator.generate_narration_scripts(
                    successful_clips, episode_info
                )

                # Validate transition count
                transitions = narration.get("transitions", [])
                expected_transitions = max(0, len(successful_clips) - 1)
                if len(transitions) != expected_transitions:
                    logger.warning(
                        f"Expected {expected_transitions} transitions but got {len(transitions)}. "
                        f"Some transitions may be missing or extra transitions will be ignored."
                    )

                # Generate TTS audio for intro
                if narration.get("intro"):
                    intro_file = temp_dir / "narration_intro.mp3"
                    intro_path = self.tts_generator.generate_speech(
                        narration["intro"], str(intro_file), label="intro"
                    )
                    if intro_path:
                        narration_files.append(("intro", intro_path))

                # Generate TTS audio for transitions
                transitions = narration.get("transitions", [])
                for i, transition_text in enumerate(transitions):
                    if transition_text:
                        transition_file = temp_dir / f"narration_transition_{i:03d}.mp3"
                        transition_path = self.tts_generator.generate_speech(
                            transition_text, str(transition_file), label=f"transition {i+1}/{len(transitions)}"
                        )
                        if transition_path:
                            narration_files.append((f"transition_{i}", transition_path))

                logger.info(f"Generated {len(narration_files)} narration segments")

            # Determine output path
            if output_path is None:
                output_path = self._get_output_path(digest_date, "outputs/audio_digests")

            # Stitch clips together with narration
            if narration_files:
                # Build interleaved file list: intro, clip1, transition1, clip2, transition2, ...
                interleaved_files = []

                # Add intro
                intro = next((path for label, path in narration_files if label == "intro"), None)
                if intro:
                    interleaved_files.append(intro)

                # Interleave clips with transitions
                for i, clip_file in enumerate(clip_files):
                    interleaved_files.append(clip_file)

                    # Add transition after clip (if not last clip)
                    if i < len(clip_files) - 1:
                        transition = next(
                            (path for label, path in narration_files if label == f"transition_{i}"),
                            None
                        )
                        if transition:
                            interleaved_files.append(transition)

                logger.info(f"Stitching {len(interleaved_files)} segments (clips + narration)...")
                final_file = self.audio_processor.stitch_audio_files(interleaved_files, output_path)
            else:
                # No narration - stitch clips only
                final_file = self.audio_processor.stitch_audio_files(clip_files, output_path)

            # Validate and save
            validation = self.audio_processor.validate_audio_output(
                final_file,
                expected_duration_min=target_duration * 0.7,
                expected_duration_max=target_duration * 1.3
            )

            self.db.add_audio_digest(
                digest_date=digest_date,
                file_path=final_file,
                duration_sec=validation.get("duration"),
                target_duration_sec=target_duration,
                clip_count=len(selected_clips),
                episode_count=len(episodes)
            )

            logger.info(f"Audio digest saved: {final_file}")
            return final_file

        finally:
            # Clean up temporary clip files
            for clip_file in clip_files:
                try:
                    if os.path.exists(clip_file):
                        os.remove(clip_file)
                except Exception:
                    pass

            # Clean up temporary narration files
            for _, narration_file in narration_files:
                try:
                    if os.path.exists(narration_file):
                        os.remove(narration_file)
                except Exception:
                    pass

            try:
                temp_dir.rmdir()
            except Exception:
                pass

    def _get_transcribed_episodes_for_date(self, date: datetime) -> List[Dict[str, Any]]:
        """Get all processed episodes for a specific date.

        Finds episodes via their summaries' digest_date, since episodes from
        different podcasts may have different publication dates but share the
        same digest processing date.

        Falls back to matching by episode publication date if no summaries
        are found for the given digest_date.

        Args:
            date: Date to query (matches summaries.digest_date)

        Returns:
            List of episode dictionaries
        """
        # Primary query: find episodes that have summaries for this digest_date
        results = self.db.conn.execute("""
            SELECT DISTINCT e.*, p.title as podcast_title
            FROM episodes e
            JOIN podcasts p ON e.podcast_id = p.id
            JOIN summaries s ON s.episode_id = e.id
            WHERE s.digest_date = ? AND e.status = 'processed'
            ORDER BY e.date DESC
        """, (date.date(),)).fetchall()

        # Fallback: if no summaries found, try matching by episode publication date
        if not results:
            results = self.db.conn.execute("""
                SELECT e.*, p.title as podcast_title
                FROM episodes e
                JOIN podcasts p ON e.podcast_id = p.id
                WHERE DATE(e.date) = ? AND e.status = 'processed'
                ORDER BY e.date DESC
            """, (date.date(),)).fetchall()

        episodes = []
        for row in results:
            episodes.append({
                "id": row[0],
                "podcast_id": row[1],
                "title": row[2],
                "date": row[3],
                "url": row[4],
                "file_path": row[5],
                "duration_seconds": row[6],
                "status": row[7],
                "created_at": row[8],
                "podcast_title": row[9]
            })

        return episodes

    def _get_episode_by_id(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Get episode by ID.

        Args:
            episode_id: Episode ID

        Returns:
            Episode dictionary or None
        """
        result = self.db.conn.execute("""
            SELECT e.*, p.title as podcast_title
            FROM episodes e
            JOIN podcasts p ON e.podcast_id = p.id
            WHERE e.id = ?
        """, (episode_id,)).fetchone()

        if result:
            return {
                "id": result[0],
                "podcast_id": result[1],
                "title": result[2],
                "date": result[3],
                "url": result[4],
                "file_path": result[5],
                "duration_seconds": result[6],
                "status": result[7],
                "created_at": result[8],
                "podcast_title": result[9]
            }
        return None

    def _estimate_episode_duration(self, episode_id: int) -> float:
        """Estimate episode duration from transcript timestamps.

        Args:
            episode_id: Episode ID

        Returns:
            Estimated duration in seconds
        """
        result = self.db.conn.execute("""
            SELECT MAX(timestamp_end) FROM transcripts WHERE episode_id = ?
        """, (episode_id,)).fetchone()

        return result[0] if result and result[0] else 0.0

    def _get_output_path(self, date: datetime, output_dir: str) -> str:
        """Generate output file path.

        Args:
            date: Date for the digest
            output_dir: Output directory

        Returns:
            Full path to output file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"audio_digest_{date.strftime('%Y-%m-%d')}.mp3"
        return str(output_path / filename)

