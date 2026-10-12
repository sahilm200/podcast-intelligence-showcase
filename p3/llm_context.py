"""Helpers for estimating token budgets and chunking long prompts."""

from typing import Dict, List

DEFAULT_CONTEXT_TOKENS = 8192
DEFAULT_RESPONSE_TOKENS = 2000
SAFETY_MARGIN_TOKENS = 256

MODEL_CONTEXT_TOKENS: Dict[str, int] = {
    "groq/compound": 32768,
    "compound": 32768,
    "compound-beta": 32768,
    "llama-3.3-70b": 131072,
    "llama-3.3-70b-versatile": 131072,
    "llama-3.1-70b": 131072,
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama-3.2-1b-preview": 131072,
    "llama-3.2-3b-preview": 131072,
    "llama-3.2-11b-preview": 131072,
    "llama-3.2-90b-preview": 131072,
    "llama-4-scout": 131072,
    "llama-4-maverick": 131072,
}


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough: ~4 chars per token)."""
    return len(text) // 4


def _model_name_variants(model_name: str) -> List[str]:
    if not model_name:
        return []
    name = model_name.lower().strip()
    variants = [name]
    if "/" in name:
        variants.append(name.split("/", 1)[1])
    return variants


def get_model_context_tokens(model_name: str, default_tokens: int = DEFAULT_CONTEXT_TOKENS) -> int:
    """Return max context tokens for a known model, or a conservative default."""
    for candidate in _model_name_variants(model_name):
        if candidate in MODEL_CONTEXT_TOKENS:
            return MODEL_CONTEXT_TOKENS[candidate]
    return default_tokens


def calculate_input_token_budget(
    model_name: str,
    prompt: str,
    desired_response_tokens: int = DEFAULT_RESPONSE_TOKENS,
    safety_margin_tokens: int = SAFETY_MARGIN_TOKENS,
    default_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> int:
    """Compute the maximum safe input tokens for a prompt + response."""
    max_context = get_model_context_tokens(model_name, default_tokens=default_context_tokens)
    prompt_tokens = estimate_tokens(prompt)
    available_tokens = max_context - prompt_tokens - desired_response_tokens - safety_margin_tokens
    return max(512, available_tokens)


def chunk_text(text: str, max_tokens: int = 4000, overlap_tokens: int = 200) -> List[str]:
    """Split text into chunks that fit within token limits."""
    if max_tokens <= 0:
        return [text]

    overlap_tokens = max(0, min(overlap_tokens, max_tokens // 2))
    max_chars = max_tokens * 4  # ~4 chars per token
    overlap_chars = overlap_tokens * 4

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end < len(text):
            search_start = end - int(max_chars * 0.2)
            for boundary in [". ", "! ", "? ", "\n\n", "\n"]:
                pos = text.rfind(boundary, search_start, end)
                if pos > search_start:
                    end = pos + len(boundary)
                    break

        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)

    return chunks

