import hashlib
import os
import uuid
from pathlib import Path
from fastapi import HTTPException

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Magic bytes signatures ────────────────────────────────────────────────────
# Format: (offset, bytes_to_match, media_type, label)
MAGIC_SIGNATURES = [
    # Images
    (0, b'\xff\xd8\xff',               "image",    "JPEG"),
    (0, b'\x89PNG\r\n\x1a\n',          "image",    "PNG"),
    (0, b'GIF87a',                      "image",    "GIF"),
    (0, b'GIF89a',                      "image",    "GIF"),
    (0, b'BM',                          "image",    "BMP"),
    (0, b'II\x2a\x00',                  "image",    "TIFF"),   # little-endian
    (0, b'MM\x00\x2a',                  "image",    "TIFF"),   # big-endian
    (0, b'RIFF',                        "image",    "WEBP"),   # needs extra check below
    # Audio
    (0, b'RIFF',                        "audio",    "WAV"),    # same as WEBP — resolved below
    (0, b'ID3',                         "audio",    "MP3"),
    (0, b'\xff\xfb',                    "audio",    "MP3"),
    (0, b'\xff\xf3',                    "audio",    "MP3"),
    (0, b'\xff\xf2',                    "audio",    "MP3"),
    (0, b'fLaC',                        "audio",    "FLAC"),
    (0, b'OggS',                        "audio",    "OGG/OPUS"),
    (4, b'ftypM4A ',                    "audio",    "M4A"),
    (4, b'ftyp',                        "audio",    "MP4/M4A"),  # resolved by ext below
    # Video
    (4, b'ftypisom',                    "video",    "MP4"),
    (4, b'ftypmp42',                    "video",    "MP4"),
    (4, b'ftypMSNV',                    "video",    "MP4"),
    (4, b'ftypM4V ',                    "video",    "M4V"),
    (0, b'\x1aE\xdf\xa3',              "video",    "MKV/WEBM"),
    (0, b'RIFF',                        "video",    "AVI"),    # resolved below
    (0, b'\x00\x00\x01\xb3',           "video",    "MPEG"),
    (0, b'\x00\x00\x01\xba',           "video",    "MPEG"),
    # Documents
    (0, b'%PDF',                        "document", "PDF"),
    (0, b'PK\x03\x04',                 "document", "DOCX/XLSX/PPTX"),  # ZIP-based Office
    (0, b'\xd0\xcf\x11\xe0',           "document", "DOC/XLS"),         # Legacy Office
]

# RIFF disambiguation (offset 8): WAV vs WEBP vs AVI
def _resolve_riff(data: bytes) -> tuple[str, str]:
    """RIFF container — check bytes 8-12 to distinguish WAV / WEBP / AVI."""
    tag = data[8:12] if len(data) >= 12 else b''
    if tag == b'WAVE':
        return "audio", "WAV"
    elif tag == b'WEBP':
        return "image", "WEBP"
    elif tag == b'AVI ':
        return "video", "AVI"
    return "unknown", "RIFF"


def validate_magic_bytes(file_bytes: bytes, declared_filename: str) -> tuple[str, str]:
    """
    Validate file magic bytes against declared extension.
    Returns (media_type, format_label).
    Raises HTTPException 415 if mismatch or unsupported.
    """
    if len(file_bytes) < 12:
        raise HTTPException(status_code=400, detail="File too small to validate.")

    declared_ext = declared_filename.rsplit(".", 1)[-1].lower() if "." in declared_filename else ""

    # Special case: RIFF container
    if file_bytes[:4] == b'RIFF':
        magic_type, fmt = _resolve_riff(file_bytes)
    else:
        magic_type, fmt = None, None
        for offset, signature, mtype, label in MAGIC_SIGNATURES:
            if file_bytes[offset:offset + len(signature)] == signature:
                # Skip RIFF entries here (handled above)
                if signature == b'RIFF':
                    continue
                magic_type, fmt = mtype, label
                break

    if magic_type is None:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported or unrecognized file format. "
                   f"Declared extension: .{declared_ext}"
        )

    # Cross-check: magic_type must agree with declared extension
    ext_type = _ext_to_type(declared_ext)
    if ext_type != "unknown" and ext_type != magic_type:
        raise HTTPException(
            status_code=415,
            detail=f"File content ({fmt}) does not match declared extension (.{declared_ext}). "
                   f"Possible file spoofing attempt."
        )

    return magic_type, fmt


def _ext_to_type(ext: str) -> str:
    image_exts  = {"jpg","jpeg","png","bmp","tiff","tif","webp","gif","heic","heif"}
    video_exts  = {"mp4","avi","mov","mkv","webm","wmv","flv","m4v"}
    audio_exts  = {"mp3","wav","ogg","flac","m4a","aac","wma","opus","aiff","aif"}
    doc_exts    = {"pdf","doc","docx","txt","rtf","odt","xls","xlsx","pptx"}
    if ext in image_exts:  return "image"
    if ext in video_exts:  return "video"
    if ext in audio_exts:  return "audio"
    if ext in doc_exts:    return "document"
    return "unknown"


# ── Existing helpers ──────────────────────────────────────────────────────────

def compute_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def detect_media_type(filename: str) -> str:
    """Detect media type by extension (used as fallback / quick check)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _ext_to_type(ext)


async def save_file_locally(file_bytes: bytes, filename: str) -> str:
    """Save locally (MVP) — production will use S3/R2."""
    unique_name = f"{uuid.uuid4()}_{filename}"
    file_path   = UPLOAD_DIR / unique_name
    file_path.write_bytes(file_bytes)
    return str(file_path)
