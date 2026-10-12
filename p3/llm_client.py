"""Shared LLM client with retry, fallback, and rate limit handling.

Provides a unified interface for making LLM calls across all P3 modules,
with support for OpenAI-compatible APIs (Groq, OpenAI, Together, Fireworks,
OpenRouter) and local Ollama.
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx

# Optional Ollama support
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# Provider base URL mapping
PROVIDER_BASE_URLS = {
    'openrouter': 'https://openrouter.ai/api/v1',
    'groq': 'https://api.groq.com/openai/v1',
    'together': 'https://api.together.xyz/v1',
    'fireworks': 'https://api.fireworks.ai/inference/v1',
    'openai': 'https://api.openai.com/v1',
    'cerebras': 'https://api.cerebras.ai/v1',  # OpenAI-compatible, free tier
    'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai',  # Gemini OpenAI-compat endpoint
}

# Environment variable mapping
PROVIDER_ENV_VARS = {
    'openrouter': 'OPENROUTER_API_KEY',
    'groq': 'GROQ_API_KEY',
    'together': 'TOGETHER_API_KEY',
    'fireworks': 'FIREWORKS_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'cerebras': 'CEREBRAS_API_KEY',
    'gemini': 'GEMINI_API_KEY',
}


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
    except Exception:
        return None


class LLMClient:
    """Unified LLM client with retry/fallback for all P3 modules.

    Supports both OpenAI-compatible cloud APIs and local Ollama.
    Handles rate limiting with exponential backoff and model fallback chains.
    """

    def __init__(self, provider: str = "ollama", model: str = "llama3.2:3b",
                 api_key: str = None, api_key_file: str = None,
                 base_url: str = None, model_fallbacks: List[str] = None,
                 ollama_model: str = None, ollama_base_url: str = "http://localhost:11434",
                 timeout: int = 120, max_retries: int = 3):
        """Initialize LLM client.

        Args:
            provider: LLM provider ('ollama', 'groq', 'openai', 'together', etc.)
            model: Primary model name
            api_key: API key for cloud providers
            api_key_file: Path to file containing API key
            base_url: Override base URL for API
            model_fallbacks: List of fallback models to try on rate limit
            ollama_model: Model to use for Ollama fallback (if provider is cloud)
            ollama_base_url: Ollama server URL
            timeout: Request timeout in seconds
            max_retries: Max retries per model before trying fallback
        """
        self.provider = provider.lower() if provider else "ollama"
        self.model = model
        self.api_key = api_key
        self.model_fallbacks = model_fallbacks or []
        self.ollama_model = ollama_model or model
        self.ollama_base_url = ollama_base_url
        self.timeout = timeout
        self.max_retries = max(1, max_retries)

        # Resolve base URL
        if base_url:
            self.base_url = base_url
        elif self.provider in PROVIDER_BASE_URLS:
            self.base_url = PROVIDER_BASE_URLS[self.provider]
        else:
            self.base_url = ollama_base_url

        # Load API key from file if needed
        if not self.api_key and api_key_file and self.provider != "ollama":
            self.api_key = load_api_key_from_file(api_key_file)

        # Load API key from environment if not provided
        if not self.api_key and self.provider != "ollama":
            env_var = PROVIDER_ENV_VARS.get(self.provider)
            if env_var:
                self.api_key = os.getenv(env_var)

    @classmethod
    def from_config(cls, llm_config: dict) -> 'LLMClient':
        """Create LLMClient from a prepare_llm_config() dict.

        Args:
            llm_config: Dict from cli.prepare_llm_config()

        Returns:
            Configured LLMClient instance
        """
        return cls(
            provider=llm_config.get('provider', 'ollama'),
            model=llm_config.get('model', 'llama3.2:3b'),
            api_key_file=llm_config.get('api_key_file'),
            base_url=llm_config.get('base_url'),
            model_fallbacks=llm_config.get('model_fallbacks'),
            ollama_model=llm_config.get('ollama_model'),
            ollama_base_url=llm_config.get('ollama_base_url', 'http://localhost:11434'),
            timeout=llm_config.get('timeout', 120),
            max_retries=llm_config.get('max_retries', 3),
        )

    @property
    def is_api(self) -> bool:
        """Whether this client uses a cloud API (not local Ollama)."""
        return self.provider != "ollama"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3,
             max_tokens: int = 2000, json_mode: bool = False) -> Optional[str]:
        """Send a chat completion request with retry/fallback.

        Args:
            messages: List of {"role": ..., "content": ...} message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            json_mode: Whether to request JSON output format

        Returns:
            Response text content, or None on failure
        """
        if self.provider == "ollama":
            return self._ollama_chat(messages, temperature, max_tokens, json_mode)

        if not self.api_key:
            # Fall back to Ollama if no API key
            if OLLAMA_AVAILABLE:
                return self._ollama_chat(messages, temperature, max_tokens, json_mode)
            return None

        return self._api_chat(messages, temperature, max_tokens, json_mode)

    def chat_json(self, messages: List[Dict[str, str]], temperature: float = 0.3,
                  max_tokens: int = 2000) -> Optional[Dict[str, Any]]:
        """Send a chat request and parse JSON response.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Parsed JSON dict, or None on failure
        """
        response = self.chat(messages, temperature, max_tokens, json_mode=True)
        if not response:
            return None

        try:
            # Extract JSON from response (handle extra text around JSON)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _api_chat(self, messages: List[Dict[str, str]], temperature: float,
                  max_tokens: int, json_mode: bool) -> Optional[str]:
        """Make API call with retry and model fallback chain."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/sahilm200/podcast-intelligence-showcase"
            headers["X-Title"] = "Parakeet Podcast Processor"

        json_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if json_mode and self.provider in ["openai", "openrouter", "groq"]:
            json_payload["response_format"] = {"type": "json_object"}

        # Build model candidate list: primary + fallbacks
        model_candidates = [self.model] + [
            m for m in self.model_fallbacks if m != self.model
        ]

        for model_index, model_name in enumerate(model_candidates):
            json_payload["model"] = model_name
            response = None
            size_error = False

            for attempt in range(self.max_retries):
                try:
                    with httpx.Client(timeout=float(self.timeout)) as client:
                        response = client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=json_payload
                        )
                except httpx.TimeoutException:
                    if attempt < self.max_retries - 1:
                        print(f"  ⚠ Request timeout, retrying...")
                        time.sleep(2 ** attempt)
                        continue
                    break
                except httpx.ConnectError:
                    print(f"  ⚠ Connection error to {self.provider}")
                    break

                if self._is_request_too_large(response):
                    print(
                        f"  ⚠ {self.provider} request too large for {model_name}. "
                        "Trying next model fallback..."
                    )
                    size_error = True
                    break

                if not self._is_rate_limit_response(response):
                    break

                # Rate limited - retry with backoff
                wait_seconds = self._parse_retry_seconds(response)
                if wait_seconds is None:
                    wait_seconds = min(3 ** attempt + 2, 30)
                wait_seconds += random.uniform(0.5, 2.0)

                if attempt < self.max_retries - 1:
                    print(
                        f"  ⚠ {self.provider} rate limited, retrying after "
                        f"{wait_seconds:.1f}s..."
                    )
                    time.sleep(wait_seconds)
                    continue

                print(f"  ⚠ {self.provider} rate limited after {self.max_retries} attempts.")

            if size_error:
                continue

            if response is not None and response.status_code == 200:
                try:
                    result = response.json()["choices"][0]["message"]["content"].strip()
                    # Brief delay after success to avoid burning through TPM limits
                    time.sleep(random.uniform(0.5, 1.5))
                    return result
                except (KeyError, IndexError, json.JSONDecodeError):
                    return None

            # If rate limited and more models to try, continue to next
            if response is not None and self._is_rate_limit_response(response):
                if model_index < len(model_candidates) - 1:
                    print(
                        f"  ⚠ {self.provider} still rate limited after retries. "
                        "Trying next model fallback..."
                    )
                continue

            # Non-rate-limit error, don't try more models
            break

        # All API models failed - try Ollama fallback
        if OLLAMA_AVAILABLE and self.ollama_model:
            print(f"  ⚠ API failed, falling back to local Ollama ({self.ollama_model})")
            return self._ollama_chat(messages, temperature, max_tokens, json_mode)

        return None

    def _ollama_chat(self, messages: List[Dict[str, str]], temperature: float,
                     max_tokens: int, json_mode: bool) -> Optional[str]:
        """Make local Ollama chat call."""
        if not OLLAMA_AVAILABLE:
            return None

        try:
            kwargs = {
                "model": self.ollama_model if self.provider != "ollama" else self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            if json_mode:
                kwargs["format"] = "json"

            response = ollama.chat(**kwargs)
            return response['message']['content'].strip()
        except Exception as e:
            print(f"  ⚠ Ollama call failed: {e}")
            return None

    @staticmethod
    def _is_rate_limit_response(response: httpx.Response) -> bool:
        """Check if response indicates rate limiting."""
        if response.status_code == 429:
            return True
        if response.status_code >= 500:
            return False
        try:
            error = response.json().get("error", {})
            msg = error.get("message", "") if isinstance(error, dict) else str(error)
            return "rate_limit" in msg.lower() or "rate limit" in msg.lower()
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _is_request_too_large(response: httpx.Response) -> bool:
        """Check if response indicates request was too large."""
        if response.status_code == 413:
            return True
        if response.status_code == 400:
            try:
                error = response.json().get("error", {})
                msg = error.get("message", "") if isinstance(error, dict) else str(error)
                return any(term in msg.lower() for term in [
                    "too large", "too many tokens", "context length",
                    "maximum context", "token limit"
                ])
            except (ValueError, AttributeError):
                pass
        return False

    @staticmethod
    def _parse_retry_seconds(response: httpx.Response) -> Optional[float]:
        """Parse retry wait time from rate limit response."""
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

        if not message:
            return None

        # Match "try again in Xm Ys" or "try again in Xs"
        match = re.search(r"try again in (\d+)m\s*(\d+(?:\.\d+)?)?s", message, re.IGNORECASE)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2)) if match.group(2) else 0
            return minutes * 60 + seconds

        match = re.search(r"try again in (\d+(?:\.\d+)?)s", message, re.IGNORECASE)
        if match:
            return float(match.group(1))

        return None

