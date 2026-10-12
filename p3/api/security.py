"""
Security utilities for P3 API
Handles encryption, CSRF protection, and other security features
"""

import logging
import secrets
import hashlib
from typing import Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import base64

from p3.api.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)


class TokenEncryption:
    """
    Encrypt and decrypt OAuth tokens for secure storage
    Uses Fernet (symmetric encryption) with key derived from JWT_SECRET_KEY
    """

    def __init__(self):
        # Derive a proper Fernet key from JWT_SECRET_KEY
        # Fernet requires exactly 32 url-safe base64-encoded bytes
        key_material = settings.JWT_SECRET_KEY.encode()
        derived_key = hashlib.sha256(key_material).digest()
        self.key = base64.urlsafe_b64encode(derived_key)
        self.cipher = Fernet(self.key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a token for storage

        Args:
            plaintext: The token to encrypt

        Returns:
            Encrypted token as a string
        """
        if not plaintext:
            return ""

        encrypted_bytes = self.cipher.encrypt(plaintext.encode())
        return encrypted_bytes.decode()

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a stored token

        Args:
            ciphertext: The encrypted token

        Returns:
            Decrypted token as a string
        """
        if not ciphertext:
            return ""

        try:
            decrypted_bytes = self.cipher.decrypt(ciphertext.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            # If decryption fails, log and return empty
            # This can happen if key changes or token is corrupted
            logger.error(f"Failed to decrypt token: {e}")
            return ""


class CSRFProtection:
    """
    CSRF protection for OAuth flows
    Generates and validates state tokens to prevent CSRF attacks
    """

    @staticmethod
    def generate_state_token() -> str:
        """
        Generate a cryptographically secure state token

        Returns:
            Random state token (32 bytes, hex encoded)
        """
        return secrets.token_hex(32)

    @staticmethod
    def encode_state(csrf_token: str, redirect_url: Optional[str] = None) -> str:
        """
        Encode CSRF token and optional redirect URL into state parameter

        Args:
            csrf_token: Random CSRF token
            redirect_url: Optional URL to redirect after auth

        Returns:
            Encoded state string
        """
        if redirect_url:
            # Format: csrf_token|redirect_url
            # We'll base64 encode to handle special characters
            state_data = f"{csrf_token}|{redirect_url}"
            encoded = base64.urlsafe_b64encode(state_data.encode()).decode()
            return encoded
        return csrf_token

    @staticmethod
    def decode_state(state: str) -> tuple[str, Optional[str]]:
        """
        Decode state parameter into CSRF token and redirect URL

        Args:
            state: Encoded state parameter

        Returns:
            Tuple of (csrf_token, redirect_url)
        """
        try:
            # Try to decode as base64
            decoded = base64.urlsafe_b64decode(state.encode()).decode()
            if "|" in decoded:
                csrf_token, redirect_url = decoded.split("|", 1)
                return csrf_token, redirect_url
            return decoded, None
        except Exception:
            # If not base64 encoded, treat as raw csrf token
            return state, None


# Singleton instances
token_encryption = TokenEncryption()
csrf_protection = CSRFProtection()


def encrypt_token(plaintext: str) -> str:
    """Convenience function to encrypt a token"""
    return token_encryption.encrypt(plaintext)


def decrypt_token(ciphertext: str) -> str:
    """Convenience function to decrypt a token"""
    return token_encryption.decrypt(ciphertext)


def generate_csrf_token() -> str:
    """Convenience function to generate CSRF token"""
    return csrf_protection.generate_state_token()


def encode_oauth_state(csrf_token: str, redirect_url: Optional[str] = None) -> str:
    """Convenience function to encode OAuth state"""
    return csrf_protection.encode_state(csrf_token, redirect_url)


def decode_oauth_state(state: str) -> tuple[str, Optional[str]]:
    """Convenience function to decode OAuth state"""
    return csrf_protection.decode_state(state)

