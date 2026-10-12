"""Generate narration scripts for audio digest using LLM."""

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


NARRATION_SYSTEM_PROMPT = """You are a professional podcast narrator creating brief, engaging transitions for an audio digest.

Your narration should be:
- **Concise**: 1-2 sentences maximum per transition
- **Professional**: Clear, informative tone suitable for business/investment audience
- **Contextual**: Briefly introduce what the next clip is about
- **Smooth**: Natural transitions that connect clips logically

For the intro, provide a brief overview (2-3 sentences) of what's covered in the digest.

Return your narration as a JSON object with this exact structure:
```json
{
  "intro": "Welcome to your audio digest. Today we're covering...",
  "transitions": [
    "Next, we hear about...",
    "Moving on to a discussion of..."
  ]
}
```

Keep each piece of narration under 15 seconds when spoken (roughly 40 words max).
"""


class NarrationGenerator:
    """Generate narration scripts using LLM.

    Supports multiple providers: Ollama (local), Groq, OpenAI, etc.
    """

    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434",
                 provider: str = "ollama", api_key: str = None, api_key_file: str = None,
                 timeout: int = 120, ollama_base_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3.2:3b", model_fallbacks: Optional[List[str]] = None):
        """Initialize narration generator.

        Args:
            model: Model to use for narration generation
            base_url: API base URL (auto-detected for known providers)
            provider: LLM provider ("ollama", "groq", "openai", etc.)
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

    def generate_narration_scripts(self, selected_clips: List[Dict[str, Any]],
                                   episode_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate intro and transition scripts for selected clips.

        Args:
            selected_clips: List of selected clips with topics
            episode_info: List of episode metadata

        Returns:
            Dictionary with 'intro' and 'transitions' keys
        """
        if not selected_clips:
            logger.warning("No clips provided for narration generation")
            return {"intro": None, "transitions": []}

        # Build user prompt
        user_prompt = self._build_narration_prompt(selected_clips, episode_info)

        try:
            # Call LLM
            narration = self._call_llm(user_prompt)
            logger.info(f"Generated narration: intro + {len(narration.get('transitions', []))} transitions")
            return narration

        except Exception as e:
            logger.error(f"Narration generation failed: {e}")
            # Fallback to simple narration
            return self._fallback_narration(selected_clips)

    def _build_narration_prompt(self, selected_clips: List[Dict[str, Any]],
                                episode_info: List[Dict[str, Any]]) -> str:
        """Build prompt for narration generation.

        Args:
            selected_clips: Selected clips
            episode_info: Episode metadata

        Returns:
            Prompt string
        """
        # Create episode lookup
        episode_map = {ep["id"]: ep for ep in episode_info}

        prompt = f"""Generate narration for an audio digest with {len(selected_clips)} clips.

CLIPS TO INTRODUCE:
"""

        for i, clip in enumerate(selected_clips):
            episode_id = clip.get("episode_id")
            episode = episode_map.get(episode_id, {})
            podcast_title = episode.get("podcast_title", "Unknown Podcast")
            episode_title = episode.get("title", "Unknown Episode")
            topic = clip.get("topic", "Discussion")

            prompt += f"\nClip {i+1}:\n"
            prompt += f"  Source: {podcast_title} - {episode_title}\n"
            prompt += f"  Topic: {topic}\n"
            if clip.get("reason"):
                prompt += f"  Why included: {clip['reason']}\n"

        prompt += f"""

TASK:
1. Write a brief intro (2-3 sentences) welcoming listeners and summarizing the digest
2. Write {len(selected_clips) - 1} transition sentences (one before each clip except the first)

Each transition should:
- Be 1-2 sentences maximum
- Briefly introduce the topic of the next clip
- Maintain professional, informative tone
- Keep under 40 words (15 seconds when spoken)

Return ONLY a JSON object with your narration. No other text.
"""

        return prompt

    def _call_llm(self, user_prompt: str) -> Dict[str, Any]:
        """Call LLM for narration generation.

        Supports multiple providers: Ollama (local), Groq, OpenAI, etc.
        Falls back to local Ollama if cloud API fails.

        Args:
            user_prompt: User prompt string

        Returns:
            Dictionary with intro and transitions
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

        return self._parse_response(response_text)

    def _call_ollama_fallback(self, user_prompt: str) -> str:
        """Call local Ollama as fallback when cloud API fails."""
        payload = {
            "model": self.ollama_model,
            "prompt": f"{NARRATION_SYSTEM_PROMPT}\n\n{user_prompt}",
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
            "prompt": f"{NARRATION_SYSTEM_PROMPT}\n\n{user_prompt}",
            "stream": False,
            "format": "json"
        }

        response = requests.post(self.api_url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    def _call_openai_compatible(self, user_prompt: str) -> str:
        """Call OpenAI-compatible API (Groq, OpenAI, etc.)."""
        if not self.api_key:
            raise ValueError(f"No API key configured for {self.provider}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        json_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": NARRATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        # Groq/OpenAI support JSON mode
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

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract narration."""
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            response_text = response_text[json_start:json_end]

        try:
            narration = json.loads(response_text)

            # Validate structure
            if "intro" not in narration:
                raise ValueError("Missing 'intro' field in narration")

            # Ensure transitions is a list
            if "transitions" not in narration:
                narration["transitions"] = []
            elif not isinstance(narration["transitions"], list):
                narration["transitions"] = []

            return narration

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Response: {response_text}")
            raise

    def _fallback_narration(self, selected_clips: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate simple fallback narration when LLM fails.

        Args:
            selected_clips: Selected clips

        Returns:
            Dictionary with intro and transitions
        """
        intro = f"Welcome to your audio digest. We've selected {len(selected_clips)} clips covering key insights from today's episodes."

        transitions = []
        for i in range(len(selected_clips) - 1):
            next_clip = selected_clips[i + 1]
            topic = next_clip.get("topic", "the next topic")
            transitions.append(f"Next, {topic}.")

        return {
            "intro": intro,
            "transitions": transitions
        }

