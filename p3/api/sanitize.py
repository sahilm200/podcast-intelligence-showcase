"""
Input sanitization utilities for P3 API
Prevents XSS and sanitizes user-submitted content
"""

import bleach
import re
from typing import Optional


# Allowed HTML tags for rich text (very restrictive)
ALLOWED_TAGS = [
    'p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li',
]

# Allowed HTML attributes
ALLOWED_ATTRIBUTES = {
    'a': ['href', 'title'],
}

# Allowed protocols for links
ALLOWED_PROTOCOLS = ['http', 'https', 'mailto']


def sanitize_html(text: str, strip: bool = False) -> str:
    """
    Sanitize HTML content to prevent XSS attacks

    Args:
        text: Input HTML text
        strip: If True, strip all HTML tags. If False, allow safe tags.

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    if strip:
        # Strip all HTML tags
        return bleach.clean(text, tags=[], strip=True)

    # Allow only safe HTML tags
    return bleach.clean(
        text,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols=ALLOWED_PROTOCOLS,
        strip=True,
    )


def sanitize_plain_text(text: str) -> str:
    """
    Sanitize plain text by stripping all HTML

    Args:
        text: Input text

    Returns:
        Plain text with no HTML
    """
    return sanitize_html(text, strip=True)


def sanitize_url(url: str) -> Optional[str]:
    """
    Sanitize and validate URL

    Args:
        url: Input URL

    Returns:
        Sanitized URL or None if invalid
    """
    if not url:
        return None

    # Remove any whitespace
    url = url.strip()

    # Basic URL validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

    if url_pattern.match(url):
        return url
    return None


def sanitize_podcast_description(description: str) -> str:
    """
    Sanitize podcast description

    Allows some basic HTML formatting but removes dangerous content

    Args:
        description: Podcast description

    Returns:
        Sanitized description
    """
    if not description:
        return ""

    # Allow basic formatting tags
    return sanitize_html(description, strip=False)


def sanitize_episode_description(description: str) -> str:
    """
    Sanitize episode description

    Allows some basic HTML formatting but removes dangerous content

    Args:
        description: Episode description

    Returns:
        Sanitized description
    """
    return sanitize_podcast_description(description)


def sanitize_user_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize general user input

    Strips all HTML and limits length

    Args:
        text: User input text
        max_length: Maximum allowed length (optional)

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Strip all HTML
    sanitized = sanitize_plain_text(text)

    # Trim length if specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized.strip()


def sanitize_email(email: str) -> Optional[str]:
    """
    Sanitize and validate email address

    Args:
        email: Email address

    Returns:
        Sanitized email or None if invalid
    """
    if not email:
        return None

    # Strip whitespace and convert to lowercase
    email = email.strip().lower()

    # Basic email validation
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    if email_pattern.match(email):
        return email
    return None


def sanitize_podcast_title(title: str, max_length: int = 200) -> str:
    """
    Sanitize podcast title

    Args:
        title: Podcast title
        max_length: Maximum length

    Returns:
        Sanitized title
    """
    return sanitize_user_input(title, max_length=max_length)


def sanitize_episode_title(title: str, max_length: int = 300) -> str:
    """
    Sanitize episode title

    Args:
        title: Episode title
        max_length: Maximum length

    Returns:
        Sanitized title
    """
    return sanitize_user_input(title, max_length=max_length)


# Convenience function for sanitizing data dict
def sanitize_podcast_data(data: dict) -> dict:
    """
    Sanitize all fields in podcast data

    Args:
        data: Podcast data dictionary

    Returns:
        Sanitized data dictionary
    """
    sanitized = data.copy()

    if 'title' in sanitized:
        sanitized['title'] = sanitize_podcast_title(sanitized['title'])

    if 'description' in sanitized:
        sanitized['description'] = sanitize_podcast_description(sanitized['description'])

    if 'rss_url' in sanitized:
        sanitized_url = sanitize_url(sanitized['rss_url'])
        if not sanitized_url:
            raise ValueError("Invalid RSS URL")
        sanitized['rss_url'] = sanitized_url

    return sanitized


def sanitize_episode_data(data: dict) -> dict:
    """
    Sanitize all fields in episode data

    Args:
        data: Episode data dictionary

    Returns:
        Sanitized data dictionary
    """
    sanitized = data.copy()

    if 'title' in sanitized:
        sanitized['title'] = sanitize_episode_title(sanitized['title'])

    if 'description' in sanitized:
        sanitized['description'] = sanitize_episode_description(sanitized['description'])

    if 'audio_url' in sanitized:
        sanitized_url = sanitize_url(sanitized['audio_url'])
        if not sanitized_url:
            raise ValueError("Invalid audio URL")
        sanitized['audio_url'] = sanitized_url

    return sanitized

