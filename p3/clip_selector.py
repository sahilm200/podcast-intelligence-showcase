"""Select best audio clips using LLM analysis."""

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import httpx

logger = logging.getLogger(__name__)


def load_api_key_from_file(file_path: str) -> Optional[str]:
    """Load API key from a file."""
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


# Provider base URL mapping
PROVIDER_BASE_URLS = {
    'openrouter': 'https://openrouter.ai/api/v1',
    'groq': 'https://api.groq.com/openai/v1',
    'together': 'https://api.together.xyz/v1',
    'fireworks': 'https://api.fireworks.ai/inference/v1',
    'openai': 'https://api.openai.com/v1',
}

# Environment variable mapping
PROVIDER_ENV_VARS = {
    'openrouter': 'OPENROUTER_API_KEY',
    'groq': 'GROQ_API_KEY',
    'together': 'TOGETHER_API_KEY',
    'fireworks': 'FIREWORKS_API_KEY',
    'openai': 'OPENAI_API_KEY',
}


CLIP_SELECTION_SYSTEM_PROMPT = """You are an expert at identifying the most valuable segments from business and tech investing podcasts.

Your task is to select audio clips that capture the BEST insights for investors and business professionals,
filling the target duration with high-quality content.

SELECTION CRITERIA (prioritize in this order):
1. **Investment insights**: Metrics, valuations, market trends, business strategies
2. **Company analysis**: Growth drivers, competitive advantages, unit economics
3. **Quotable moments**: Compelling statements with specific numbers or frameworks
4. **Self-contained**: The clip makes sense without additional context
5. **Distinct insights**: Each clip should cover different topics (no redundancy)
6. **Action-oriented**: Insights that inform investment decisions or business strategy

SPEAKER GUIDANCE (when speaker labels are available):
- Prefer clips where a GUEST speaker delivers complete insights (hosts often just introduce topics)
- Single-speaker clips are often cleaner than rapid back-and-forth dialogue
- If a clip has multiple speakers, ensure it captures a complete exchange, not a fragment

WHAT TO AVOID:
- Generic statements without specifics
- Introduction/conclusion fluff
- Questions without answers (unless the question itself is insightful)
- Mid-sentence or incomplete thoughts
- Redundant information across clips
- Clips that cut off mid-conversation

APPROACH:
- Select enough clips to fill the target duration
- Prioritize quality over quantity, but ensure substantial content coverage
- Score each clip **0-100** based on value to investors
- Respect the constraints provided in the user prompt

Return your selection as a JSON array with this exact structure:
```json
[
  {
    "candidate_index": 0,
    "episode_id": 123,
    "start_sec": 456.7,
    "end_sec": 489.2,
    "topic": "Brief description of what this clip covers",
    "reason": "Why this clip is valuable for investors",
    "score": 85
  }
]
```

Be selective. If there aren't enough high-quality clips, return fewer clips rather than including mediocre content.
"""


class ClipSelector:
    """Uses LLM to select the best clips from candidates.

    Supports multiple LLM providers:
    - Ollama (local, default)
    - Groq (cloud, fast, free tier)
    - OpenAI (cloud)
    - Together, Fireworks, OpenRouter (cloud)
    """

    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434",
                 provider: str = "ollama", api_key: str = None, api_key_file: str = None,
                 timeout: int = 120, ollama_base_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3.2:3b", model_fallbacks: Optional[List[str]] = None):
        """Initialize clip selector.

        Args:
            model: Model to use for selection
            base_url: API base URL (auto-detected for known providers)
            provider: LLM provider ("ollama", "groq", "openai", "together", etc.)
            api_key: API key for cloud providers
            api_key_file: Path to file containing API key
            timeout: Request timeout in seconds
            ollama_base_url: Ollama base URL for fallback when API fails
            ollama_model: Ollama model to use for fallback
        """
        self.model = model
        self.provider = provider.lower()
        self.timeout = timeout
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.model_fallbacks = model_fallbacks or []

        # Determine base URL
        if self.provider in PROVIDER_BASE_URLS:
            self.base_url = PROVIDER_BASE_URLS[self.provider]
        else:
            self.base_url = base_url

        # For Ollama, use the generate endpoint
        if self.provider == "ollama":
            self.api_url = f"{base_url}/api/generate"
        else:
            self.api_url = f"{self.base_url}/chat/completions"

        # Load API key from file if specified
        if not self.api_key and api_key_file and self.provider != "ollama":
            self.api_key = load_api_key_from_file(api_key_file)
            if self.api_key:
                logger.info(f"Loaded API key from {api_key_file}")

        # Load API key from environment if not provided
        if not self.api_key and self.provider != "ollama":
            env_var = PROVIDER_ENV_VARS.get(self.provider)
            if env_var:
                self.api_key = os.getenv(env_var)
                if self.api_key:
                    logger.info(f"Loaded API key from environment variable {env_var}")

    def select_clips(self, candidates: List[Dict[str, Any]],
                    episode_info: List[Dict[str, Any]],
                    target_duration: int = 60,
                    max_clips: int = 14,
                    max_clips_per_episode: int = 20,
                    min_clip_duration: float = 30.0,
                    max_clip_duration: float = 180.0) -> List[Dict[str, Any]]:
        """Select best clips from candidates using LLM.

        Args:
            candidates: List of candidate clips with episode_id, start_sec, end_sec, duration, text
            episode_info: List of episode metadata (id, title, podcast_title)
            target_duration: Target total duration for all clips in seconds
            max_clips: Maximum number of clips to select
            max_clips_per_episode: Maximum clips allowed from a single episode
            min_clip_duration: Minimum clip duration in seconds
            max_clip_duration: Maximum clip duration in seconds

        Returns:
            List of selected clips with enriched metadata
        """
        if not candidates:
            logger.warning("No candidates provided for clip selection")
            return []

        # Build user prompt
        user_prompt = self._build_user_prompt(
            candidates, episode_info, target_duration, max_clips,
            max_clips_per_episode, min_clip_duration, max_clip_duration
        )

        # Call Ollama
        try:
            selected_clips = self._call_llm(user_prompt)

            # Enrich selected clips with full candidate data
            enriched_clips = self._enrich_clips(selected_clips, candidates)

            logger.info(f"Selected {len(enriched_clips)} clips from {len(candidates)} candidates")
            return enriched_clips

        except Exception as e:
            logger.error(f"LLM clip selection failed: {e}")
            # Fallback: select first N clips with per-episode limits
            logger.warning(f"Falling back to first {max_clips} candidates")
            return self._fallback_selection(candidates, max_clips, max_clips_per_episode)

    def _build_user_prompt(self, candidates: List[Dict[str, Any]],
                          episode_info: List[Dict[str, Any]],
                          target_duration: int,
                          max_clips: int,
                          max_clips_per_episode: int,
                          min_clip_duration: float,
                          max_clip_duration: float) -> str:
        """Build user prompt for LLM.

        Args:
            candidates: List of candidate clips
            episode_info: List of episode metadata
            target_duration: Target total duration
            max_clips: Maximum clips to select
            max_clips_per_episode: Maximum clips from any single episode
            min_clip_duration: Minimum clip duration
            max_clip_duration: Maximum clip duration

        Returns:
            User prompt string
        """
        # Create episode lookup
        episode_map = {ep["id"]: ep for ep in episode_info}

        prompt = f"""Select the best clips for a {target_duration}-second audio digest.

TARGET DURATION: ~{target_duration} seconds total
AVAILABLE CANDIDATES: {len(candidates)}
GOAL: Select enough clips to fill the target duration with the highest-quality content

"""

        # Group candidates by episode
        by_episode = {}
        for i, candidate in enumerate(candidates):
            ep_id = candidate.get("episode_id")
            if ep_id not in by_episode:
                by_episode[ep_id] = []
            by_episode[ep_id].append((i, candidate))

        # Format candidates by episode
        for ep_id, clips in by_episode.items():
            episode = episode_map.get(ep_id, {})
            podcast_title = episode.get("podcast_title", "Unknown Podcast")
            episode_title = episode.get("title", "Unknown Episode")

            prompt += f"\n## {podcast_title} - {episode_title}\n"
            prompt += f"Episode ID: {ep_id}\n\n"

            for candidate_idx, clip in clips:
                duration = clip["duration"]
                text = clip["text"][:300]  # Truncate long text
                if len(clip["text"]) > 300:
                    text += "..."

                prompt += f"**Candidate {candidate_idx}** ({duration:.1f}s)\n"
                prompt += f"Timestamp: {clip['start_sec']:.1f}s - {clip['end_sec']:.1f}s\n"

                # Include speaker info if available
                speaker = clip.get("speaker")
                speaker_count = clip.get("speaker_count", 0)
                if speaker:
                    if speaker_count > 1:
                        prompt += f"Speaker: {speaker} (primary, {speaker_count} speakers total)\n"
                    else:
                        prompt += f"Speaker: {speaker}\n"

                prompt += f"Text: {text}\n\n"

        prompt += f"""
Now select up to {max_clips} BEST clips that would be most valuable for investors and business professionals.

CONSTRAINTS:
- Target {target_duration} seconds total duration
- Each clip: {min_clip_duration}-{max_clip_duration} seconds
- Maximum {max_clips_per_episode} clips from any single episode
- Focus on investment insights, metrics, strategies, and quotable moments
- Each clip should be self-contained and distinct

SELECTION STRATEGY:
Select enough clips to approach the target duration. For a {target_duration}s target, you should select
approximately {int(target_duration / ((min_clip_duration + max_clip_duration) / 2))} clips on average.
Prioritize quality over quantity, but ensure you're providing substantial content.

Return ONLY a JSON array with your selections. No other text.
"""

        return prompt

    def _call_llm(self, user_prompt: str) -> List[Dict[str, Any]]:
        """Call LLM for clip selection.

        Supports multiple providers: Ollama (local), Groq, OpenAI, etc.
        Falls back to local Ollama if cloud API fails.

        Args:
            user_prompt: User prompt string

        Returns:
            List of selected clips from LLM response
        """
        if self.provider == "ollama":
            response_text = self._call_ollama(user_prompt)
        else:
            try:
                response_text = self._call_openai_compatible(user_prompt)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str or "413" in error_str:
                    logger.warning(f"{self.provider} API rate limited, falling back to local Ollama")
                    print(f"  ⚠ {self.provider} API rate limited, falling back to local Ollama...")
                    response_text = self._call_ollama_fallback(user_prompt)
                else:
                    raise

        return self._parse_llm_response(response_text)

    def _call_ollama_fallback(self, user_prompt: str) -> str:
        """Call local Ollama as fallback when cloud API fails."""
        payload = {
            "model": self.ollama_model,
            "prompt": f"{CLIP_SELECTION_SYSTEM_PROMPT}\n\n{user_prompt}",
            "stream": False,
            "format": "json"
        }

        ollama_url = f"{self.ollama_base_url}/api/generate"
        response = requests.post(ollama_url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    def _call_ollama(self, user_prompt: str) -> str:
        """Call Ollama local LLM."""
        payload = {
            "model": self.model,
            "prompt": f"{CLIP_SELECTION_SYSTEM_PROMPT}\n\n{user_prompt}",
            "stream": False,
            "format": "json"
        }

        response = requests.post(self.api_url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    def _call_openai_compatible(self, user_prompt: str) -> str:
        """Call OpenAI-compatible API (Groq, OpenAI, Together, etc.)."""
        if not self.api_key:
            raise ValueError(f"No API key configured for {self.provider}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # OpenRouter requires additional headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/parakeet-podcast-processor"
            headers["X-Title"] = "Parakeet Podcast Processor"

        json_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": CLIP_SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }

        # Groq supports JSON mode
        if self.provider in ["groq", "openai"]:
            json_payload["response_format"] = {"type": "json_object"}

        model_candidates = [self.model] + [m for m in self.model_fallbacks if m != self.model]
        last_error = None

        max_retries = 3
        for model_index, model_name in enumerate(model_candidates):
            json_payload["model"] = model_name
            response = None
            size_error = False
            for attempt in range(max_retries):
                with httpx.Client(timeout=float(self.timeout)) as client:
                    response = client.post(
                        self.api_url,
                        headers=headers,
                        json=json_payload
                    )

                if self._is_request_too_large(response):
                    print(
                        f"  ⚠ {self.provider} request too large for {model_name}. "
                        "Trying next model fallback..."
                    )
                    size_error = True
                    break
                if not self._is_rate_limit_response(response):
                    break

                wait_seconds = self._parse_retry_seconds(response)
                if wait_seconds is None:
                    wait_seconds = min(3 ** attempt + 2, 30)
                wait_seconds += random.uniform(0.5, 2.0)
                if attempt < max_retries - 1:
                    print(
                        f"  ⚠ {self.provider} rate limited, retrying after "
                        f"{wait_seconds:.1f}s..."
                    )
                    time.sleep(wait_seconds)
                    continue
                print(
                    f"  ⚠ {self.provider} rate limited after {max_retries} attempts."
                )

            if size_error:
                continue
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()

            if self._is_rate_limit_response(response):
                last_error = response
                if model_index < len(model_candidates) - 1:
                    print(
                        f"  ⚠ {self.provider} still rate limited after retries. "
                        "Trying next model fallback..."
                    )
                continue

            raise Exception(f"{self.provider} API error: {response.status_code} - {response.text}")

        if last_error is not None:
            raise Exception(f"{self.provider} API error: {last_error.status_code} - {last_error.text}")
        raise Exception(f"{self.provider} API error: no response")

    def _parse_retry_seconds(self, response: httpx.Response) -> Optional[int]:
        """Parse retry seconds from API rate limit responses."""
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

    @staticmethod
    def _is_rate_limit_response(response: httpx.Response) -> bool:
        if response.status_code == 429:
            return True
        try:
            error = response.json().get("error", {})
        except (ValueError, AttributeError, TypeError):
            return False
        code = str(error.get("code", "")).lower()
        message = str(error.get("message", "")).lower()
        if "rate_limit" in code:
            return True
        return "tokens per minute" in message or "rate limit" in message

    @staticmethod
    def _is_request_too_large(response: httpx.Response) -> bool:
        try:
            message = response.json().get("error", {}).get("message", "")
        except (ValueError, AttributeError, TypeError):
            return False
        lowered = message.lower()
        return (
            "request too large" in lowered
            or "context length" in lowered
            or "maximum context" in lowered
        )

    def _parse_llm_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract selected clips.

        Args:
            response_text: Raw text response from LLM

        Returns:
            List of selected clips
        """
        # Extract JSON from response (in case there's extra text)
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1

        # Also try to find object format (some models wrap in {"clips": [...]}
        if json_start < 0:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            response_text = response_text[json_start:json_end]

        # Parse JSON response
        try:
            selected = json.loads(response_text)
            if not isinstance(selected, list):
                # If it's a dict with a key like "clips" or "selections", extract it
                for key in ["clips", "selections", "selected_clips"]:
                    if key in selected and isinstance(selected[key], list):
                        selected = selected[key]
                        break

            # Validate structure
            if not isinstance(selected, list):
                raise ValueError("Response is not a list")

            # Validate each clip has required fields
            for clip in selected:
                required_fields = ["candidate_index"]
                for field in required_fields:
                    if field not in clip:
                        raise ValueError(f"Missing required field: {field}")

                # Validate optional but expected fields
                if "score" in clip and not isinstance(clip["score"], (int, float)):
                    raise ValueError(f"Invalid score type: {type(clip['score'])}")

            return selected

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Response: {response_text}")
            raise

    def _enrich_clips(self, selected_clips: List[Dict[str, Any]],
                     candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich selected clips with full candidate data.

        Args:
            selected_clips: Clips selected by LLM (with candidate_index)
            candidates: Original candidate clips

        Returns:
            Enriched clips with full metadata
        """
        enriched = []

        for selected in selected_clips:
            candidate_idx = selected.get("candidate_index")
            # Check for None, negative, or out-of-bounds indices
            if candidate_idx is None or candidate_idx < 0 or candidate_idx >= len(candidates):
                logger.warning(f"Invalid candidate_index: {candidate_idx} (valid range: 0-{len(candidates)-1})")
                continue

            # Get full candidate data
            candidate = candidates[candidate_idx]

            # Merge LLM selection with candidate data
            enriched_clip = {
                **candidate,  # Original candidate data (start_sec, end_sec, text, etc.)
                "topic": selected.get("topic"),
                "reason": selected.get("reason"),
                "score": selected.get("score")
            }

            enriched.append(enriched_clip)

        return enriched

    def _fallback_selection(self, candidates: List[Dict[str, Any]],
                           max_clips: int,
                           max_clips_per_episode: int = 20) -> List[Dict[str, Any]]:
        """Fallback clip selection when LLM fails.

        Selects clips to fill max_clips, with dynamic per-episode limits based on
        the number of unique episodes available and the configured max_clips_per_episode.

        Args:
            candidates: List of candidate clips
            max_clips: Maximum total clips to select
            max_clips_per_episode: Maximum clips allowed from a single episode (from config)

        Returns:
            Selected candidates with default scores
        """
        selected = []

        # Count unique episodes in candidates
        unique_episodes = set(c.get("episode_id") for c in candidates)
        num_episodes = len(unique_episodes)

        # Calculate dynamic per-episode limit:
        # - Distribute clips evenly across episodes
        # - Minimum of 2 clips per episode (to get meaningful content)
        # - Respect the configured max_clips_per_episode ceiling
        dynamic_limit = max(2, (max_clips // max(num_episodes, 1)) + 1)
        clips_per_episode = min(dynamic_limit, max_clips_per_episode)

        logger.info(f"Fallback selection: {num_episodes} episodes, {clips_per_episode} clips/episode limit (max {max_clips_per_episode}), {max_clips} total target")
        print(f"  Fallback: selecting up to {max_clips} clips ({clips_per_episode}/episode, {num_episodes} episodes)")

        for candidate in candidates:
            if len(selected) >= max_clips:
                break

            episode_id = candidate.get("episode_id")

            # Check current count for this episode
            episode_count = sum(1 for c in selected if c.get("episode_id") == episode_id)
            if episode_count >= clips_per_episode:
                continue

            selected.append({
                **candidate,
                "topic": "Auto-selected clip",
                "reason": "Fallback selection (LLM unavailable)",
                "score": 50
            })

        logger.info(f"Fallback selected {len(selected)} clips")
        return selected

