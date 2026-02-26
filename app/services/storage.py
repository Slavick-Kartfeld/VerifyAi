import hashlib
import os
import uuid
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def compute_sha256(file_bytes: bytes) -> str:
    """חישוב SHA-256 לשמירת chain of custody"""
    return hashlib.sha256(file_bytes).hexdigest()


def detect_media_type(filename: str) -> str:
    """זיהוי סוג מדיה לפי סיומת"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    image_exts = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
    video_exts = {"mp4", "avi", "mov", "mkv", "webm"}
    audio_exts = {"mp3", "wav", "ogg", "flac", "m4a"}
    doc_exts = {"pdf", "doc", "docx", "txt", "tif", "tiff"}

    if ext in image_exts:
        return "image"
    elif ext in video_exts:
        return "video"
    elif ext in audio_exts:
        return "audio"
    elif ext in doc_exts:
        return "document"
    return "unknown"


async def save_file_locally(file_bytes: bytes, filename: str) -> str:
    """שמירה מקומית (MVP) — בפרודקשן יעבור ל-S3"""
    unique_name = f"{uuid.uuid4()}_{filename}"
    file_path = UPLOAD_DIR / unique_name
    file_path.write_bytes(file_bytes)
    return str(file_path)
