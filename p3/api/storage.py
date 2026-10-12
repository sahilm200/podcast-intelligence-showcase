"""
Storage utilities for P3 API
Handles temporary audio storage and clip management with Cloudflare R2
FIXED: Proper async file handling, correct R2 validation, integrated with worker
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import httpx
import aiofiles
import boto3
from botocore.exceptions import ClientError

from p3.api.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)


class TemporaryAudioStorage:
    """
    Manage temporary audio file storage for processing
    Files are automatically deleted after processing
    """

    def __init__(self, base_dir: str = "/tmp/p3-episodes"):
        self.base_dir = Path(base_dir)
        # Create directory if it doesn't exist
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create temp dir {base_dir}, using system temp: {e}")
            # Fallback to system temp
            import tempfile
            self.base_dir = Path(tempfile.gettempdir()) / "p3-episodes"
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, episode_id: str) -> Path:
        """Get temporary file path for episode"""
        return self.base_dir / f"{episode_id}.mp3"

    async def download(self, episode_id: str, audio_url: str) -> Path:
        """
        Download audio file temporarily using async operations

        Args:
            episode_id: Episode UUID
            audio_url: URL to download from

        Returns:
            Path to downloaded file
        """
        file_path = self.get_path(episode_id)

        try:
            logger.info(f"Downloading episode {episode_id} from {audio_url}")

            # Stream in 64KB chunks to avoid loading the entire audio file into RAM.
            # A 200MB episode would otherwise consume 200MB per concurrent download.
            bytes_downloaded = 0
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
                async with client.stream('GET', audio_url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            await f.write(chunk)
                            bytes_downloaded += len(chunk)

            logger.info(f"Downloaded {bytes_downloaded:,} bytes to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to download episode {episode_id}: {e}", exc_info=True)
            # Clean up partial file if exists
            if file_path.exists():
                file_path.unlink()
            raise

    def exists(self, episode_id: str) -> bool:
        """Check if temporary file exists"""
        return self.get_path(episode_id).exists()

    def delete(self, episode_id: str) -> bool:
        """
        Delete temporary audio file

        Args:
            episode_id: Episode UUID

        Returns:
            True if deleted, False if not found
        """
        file_path = self.get_path(episode_id)

        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted temporary file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False

    async def cleanup_old_files(self, max_age_hours: int = 1) -> int:
        """
        Delete temporary files older than max_age_hours

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        deleted_count = 0

        try:
            for file_path in self.base_dir.glob("*.mp3"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Cleaned up old file: {file_path.name}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old temporary files")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        return deleted_count


class CloudflareR2Storage:
    """
    Manage clip storage in Cloudflare R2
    Uses S3-compatible API via boto3
    """

    def __init__(self):
        # Check if R2 is properly configured (non-empty values)
        self.enabled = all([
            settings.R2_ACCOUNT_ID and settings.R2_ACCOUNT_ID != "",
            settings.R2_ACCESS_KEY_ID and settings.R2_ACCESS_KEY_ID != "",
            settings.R2_SECRET_ACCESS_KEY and settings.R2_SECRET_ACCESS_KEY != "",
            settings.R2_BUCKET_NAME and settings.R2_BUCKET_NAME != "",
        ])

        if self.enabled:
            try:
                # Initialize S3 client for R2
                self.client = boto3.client(
                    's3',
                    endpoint_url=f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
                    aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                    region_name='auto',
                )
                self.bucket_name = settings.R2_BUCKET_NAME
                logger.info(f"R2 storage initialized: bucket={self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to initialize R2 client: {e}")
                self.enabled = False
                self.client = None
                self.bucket_name = None
        else:
            logger.warning("R2 storage not configured - clips will not be stored in cloud")
            self.client = None
            self.bucket_name = None

    async def upload_clip(
        self,
        episode_id: str,
        clip_id: str,
        clip_data: bytes,
        content_type: str = "audio/mpeg"
    ) -> Optional[str]:
        """
        Upload audio clip to R2

        Args:
            episode_id: Episode UUID
            clip_id: Clip UUID
            clip_data: Audio data bytes
            content_type: MIME type

        Returns:
            Public URL or None if upload failed
        """
        if not self.enabled:
            logger.debug("R2 not configured - skipping upload")
            return None

        # Generate object key: clips/{episode_id}/{clip_id}.mp3
        key = f"clips/{episode_id}/{clip_id}.mp3"

        try:
            # Upload to R2 (wrapped in thread since boto3 is sync)
            await asyncio.to_thread(
                self.client.put_object,
                Bucket=self.bucket_name,
                Key=key,
                Body=clip_data,
                ContentType=content_type,
                CacheControl='public, max-age=31536000, immutable',
            )

            logger.info(f"Uploaded clip {clip_id} to R2: {key} ({len(clip_data)} bytes)")

            # Return URL (use custom domain if configured)
            if settings.R2_PUBLIC_URL:
                return f"{settings.R2_PUBLIC_URL}/{key}"
            else:
                return f"https://{self.bucket_name}.r2.cloudflarestorage.com/{key}"

        except ClientError as e:
            logger.error(f"Failed to upload clip to R2: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading to R2: {e}", exc_info=True)
            return None

    async def generate_signed_url(
        self,
        episode_id: str,
        clip_id: str,
        expires_in: int = 86400
    ) -> Optional[str]:
        """
        Generate signed URL for private clip access

        Args:
            episode_id: Episode UUID
            clip_id: Clip UUID
            expires_in: Expiration time in seconds (default 24 hours)

        Returns:
            Signed URL or None if failed
        """
        if not self.enabled:
            return None

        key = f"clips/{episode_id}/{clip_id}.mp3"

        try:
            url = await asyncio.to_thread(
                self.client.generate_presigned_url,
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': key,
                },
                ExpiresIn=expires_in,
            )

            return url

        except ClientError as e:
            logger.error(f"Failed to generate signed URL: {e}")
            return None

    async def delete_clip(self, episode_id: str, clip_id: str) -> bool:
        """
        Delete clip from R2

        Args:
            episode_id: Episode UUID
            clip_id: Clip UUID

        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled:
            return False

        key = f"clips/{episode_id}/{clip_id}.mp3"

        try:
            await asyncio.to_thread(
                self.client.delete_object,
                Bucket=self.bucket_name,
                Key=key,
            )

            logger.info(f"Deleted clip from R2: {key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to delete clip from R2: {e}")
            return False

    async def delete_episode_clips(self, episode_id: str) -> int:
        """
        Delete all clips for an episode

        Args:
            episode_id: Episode UUID

        Returns:
            Number of clips deleted
        """
        if not self.enabled:
            return 0

        prefix = f"clips/{episode_id}/"
        deleted_count = 0

        try:
            # List all clips for episode
            response = await asyncio.to_thread(
                self.client.list_objects_v2,
                Bucket=self.bucket_name,
                Prefix=prefix,
            )

            if 'Contents' in response:
                for obj in response['Contents']:
                    await asyncio.to_thread(
                        self.client.delete_object,
                        Bucket=self.bucket_name,
                        Key=obj['Key'],
                    )
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} clips for episode {episode_id}")
            return deleted_count

        except ClientError as e:
            logger.error(f"Failed to delete episode clips: {e}")
            return 0


# Singleton instances
temp_storage = TemporaryAudioStorage()
r2_storage = CloudflareR2Storage()


# Helper functions for easy access
async def download_episode_temp(episode_id: str, audio_url: str) -> Path:
    """Download episode to temporary storage"""
    return await temp_storage.download(episode_id, audio_url)


def delete_episode_temp(episode_id: str) -> bool:
    """Delete temporary episode file"""
    return temp_storage.delete(episode_id)


async def upload_clip_to_r2(episode_id: str, clip_id: str, clip_data: bytes) -> Optional[str]:
    """Upload clip to R2 storage"""
    return await r2_storage.upload_clip(episode_id, clip_id, clip_data)


async def cleanup_temp_files(max_age_hours: int = 1) -> int:
    """Clean up old temporary files"""
    return await temp_storage.cleanup_old_files(max_age_hours)

