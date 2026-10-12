"""Generate candidate audio clips from podcast transcripts."""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ClipGenerator:
    """Generates candidate audio clips from transcript segments."""

    def __init__(self, min_duration: float = 15.0, max_duration: float = 45.0):
        """Initialize clip generator.

        Args:
            min_duration: Minimum clip duration in seconds
            max_duration: Maximum clip duration in seconds
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sentence_boundaries = {'.', '!', '?', ';'}

    def generate_candidates(self, transcript_segments: List[Dict[str, Any]],
                          max_candidates: int = 30) -> List[Dict[str, Any]]:
        """Generate candidate clips from transcript segments.

        Groups consecutive segments into clips of 15-45 seconds, attempting
        to avoid mid-sentence breaks using punctuation detection.
        When speaker labels are available, also uses speaker changes as
        natural clip boundaries.

        Args:
            transcript_segments: List of transcript segments from database
                Each segment should have: timestamp_start, timestamp_end, text
                Optional: speaker (for speaker-aware clipping)
            max_candidates: Maximum number of candidates to generate

        Returns:
            List of candidate clips, each with:
            - start_sec: float
            - end_sec: float
            - duration: float
            - text: str
            - segment_indices: List[int] (for debugging)
            - speaker: str or None (dominant speaker if available)
            - speaker_count: int (number of unique speakers in clip)
        """
        if not transcript_segments:
            return []

        candidates = []
        current_clip = {
            "segments": [],
            "start_sec": None,
            "end_sec": None,
            "text_parts": [],
            "speakers": []  # Track speakers for each segment
        }

        for i, segment in enumerate(transcript_segments):
            # Initialize first clip
            if current_clip["start_sec"] is None:
                current_clip["start_sec"] = segment["timestamp_start"]

            # Check for speaker change (if speaker labels available)
            current_speaker = segment.get("speaker")
            previous_speaker = current_clip["speakers"][-1] if current_clip["speakers"] else None

            # Calculate current duration before adding this segment
            if current_clip["end_sec"] is not None:
                duration = current_clip["end_sec"] - current_clip["start_sec"]
            else:
                duration = 0

            # Speaker change can trigger clip finalization if we have enough content
            speaker_changed = (
                current_speaker is not None and
                previous_speaker is not None and
                current_speaker != previous_speaker
            )

            # If speaker changes and we have minimum duration, consider finalizing
            if speaker_changed and duration >= self.min_duration:
                # Finalize current clip before starting new one with new speaker
                candidate = self._finalize_clip(current_clip, transcript_segments)
                if candidate:
                    candidates.append(candidate)

                # Start new clip with this segment
                current_clip = {
                    "segments": [],
                    "start_sec": segment["timestamp_start"],
                    "end_sec": None,
                    "text_parts": [],
                    "speakers": []
                }

                if len(candidates) >= max_candidates:
                    break

            # Add segment to current clip
            current_clip["segments"].append(i)
            current_clip["end_sec"] = segment["timestamp_end"]
            current_clip["text_parts"].append(segment["text"].strip())
            current_clip["speakers"].append(current_speaker)

            # Calculate current duration
            duration = current_clip["end_sec"] - current_clip["start_sec"]

            # Check if we should finalize this clip
            should_finalize = False

            if duration >= self.min_duration:
                # If we're in the valid range and hit a sentence boundary, finalize
                if self._ends_with_sentence_boundary(segment["text"]):
                    should_finalize = True
                # If we exceed max duration, finalize even without sentence boundary
                elif duration >= self.max_duration:
                    should_finalize = True

            # If exceeded max by a lot, backtrack to find sentence boundary
            if duration > self.max_duration * 1.2:
                # Find last sentence boundary in accumulated segments
                backtrack_index = self._find_last_sentence_boundary(current_clip["text_parts"])
                if backtrack_index is not None and backtrack_index > 0:
                    # Trim to last sentence boundary
                    current_clip["segments"] = current_clip["segments"][:backtrack_index + 1]
                    current_clip["text_parts"] = current_clip["text_parts"][:backtrack_index + 1]
                    current_clip["speakers"] = current_clip["speakers"][:backtrack_index + 1]
                    last_seg_idx = current_clip["segments"][-1]
                    current_clip["end_sec"] = transcript_segments[last_seg_idx]["timestamp_end"]
                should_finalize = True

            if should_finalize:
                # Create candidate from current clip
                candidate = self._finalize_clip(current_clip, transcript_segments)
                if candidate:
                    candidates.append(candidate)

                # Start new clip from next segment
                current_clip = {
                    "segments": [],
                    "start_sec": None,
                    "end_sec": None,
                    "text_parts": [],
                    "speakers": []
                }

                # Stop if we've reached max candidates
                if len(candidates) >= max_candidates:
                    break

        # Add final clip if it meets minimum duration
        if current_clip["start_sec"] is not None:
            duration = current_clip["end_sec"] - current_clip["start_sec"]
            if duration >= self.min_duration:
                candidate = self._finalize_clip(current_clip, transcript_segments)
                if candidate:
                    candidates.append(candidate)

        logger.info(f"Generated {len(candidates)} candidate clips from {len(transcript_segments)} segments")
        return candidates

    def _ends_with_sentence_boundary(self, text: str) -> bool:
        """Check if text ends with a sentence boundary."""
        text = text.strip()
        return any(text.endswith(char) for char in self.sentence_boundaries)

    def _find_last_sentence_boundary(self, text_parts: List[str]) -> int:
        """Find index of last text part that ends with sentence boundary.

        Args:
            text_parts: List of text segments

        Returns:
            Index of last segment with sentence boundary, or None
        """
        for i in range(len(text_parts) - 1, -1, -1):
            if self._ends_with_sentence_boundary(text_parts[i]):
                return i
        return None

    def _finalize_clip(self, clip_data: Dict[str, Any],
                       transcript_segments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert accumulated clip data to final candidate format.

        Args:
            clip_data: Dictionary with segments, start_sec, end_sec, text_parts, speakers
            transcript_segments: Original transcript segments (for backwards compatibility)

        Returns:
            Candidate clip dictionary or None if invalid
        """
        if not clip_data["segments"]:
            return None

        start_sec = clip_data["start_sec"]
        end_sec = clip_data["end_sec"]
        duration = end_sec - start_sec

        # Validate duration
        if duration < self.min_duration or duration > self.max_duration * 1.5:
            return None

        # Combine text parts
        text = " ".join(clip_data["text_parts"])

        # Determine speaker info
        speakers = clip_data.get("speakers", [])
        unique_speakers = set(s for s in speakers if s is not None)
        speaker_count = len(unique_speakers)

        # Find dominant speaker (most frequent)
        dominant_speaker = None
        if speakers:
            from collections import Counter
            speaker_counts = Counter(s for s in speakers if s is not None)
            if speaker_counts:
                dominant_speaker = speaker_counts.most_common(1)[0][0]

        return {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": duration,
            "text": text,
            "segment_indices": clip_data["segments"],
            "speaker": dominant_speaker,
            "speaker_count": speaker_count
        }

    def validate_clip_selection(self, selected_clips: List[Dict[str, Any]],
                               max_clips_per_episode: int = 20) -> Tuple[bool, str]:
        """Validate selected clips meet constraints.

        Args:
            selected_clips: List of selected clips (should have episode_id, start_sec, end_sec)
            max_clips_per_episode: Maximum clips allowed per episode

        Returns:
            (is_valid, error_message) tuple
        """
        # Check duration constraints
        for clip in selected_clips:
            duration = clip["end_sec"] - clip["start_sec"]
            if duration < self.min_duration:
                return False, f"Clip too short: {duration:.1f}s (min {self.min_duration}s)"
            if duration > self.max_duration:
                return False, f"Clip too long: {duration:.1f}s (max {self.max_duration}s)"

        # Check max clips per episode
        episode_counts = {}
        for clip in selected_clips:
            episode_id = clip.get("episode_id")
            if episode_id:
                episode_counts[episode_id] = episode_counts.get(episode_id, 0) + 1

        for episode_id, count in episode_counts.items():
            if count > max_clips_per_episode:
                return False, f"Too many clips from episode {episode_id}: {count} (max {max_clips_per_episode})"

        return True, ""

