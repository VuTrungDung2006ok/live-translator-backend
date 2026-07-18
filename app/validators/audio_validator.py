from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_LANGUAGE_HINTS,
)


# Browsers and operating systems do not always send the same
# MIME type for the same audio format.
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",

    "audio/mpeg",
    "audio/mp3",

    "audio/mp4",
    "audio/x-m4a",

    "audio/webm",
    "video/webm",

    "audio/ogg",
    "application/ogg",

    "video/mp4",
    "video/quicktime",

    # Swagger, browsers, or some clients may use this
    # even when the filename has a valid audio extension.
    "application/octet-stream",
}


def get_audio_extension(filename: str) -> str:
    """
    Return the lowercase file extension, including the dot.

    Example:
        recording.WAV -> .wav
    """

    extension = Path(filename).suffix.lower()

    if not extension:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "missing_audio_extension",
                "message": (
                    "The uploaded file does not have "
                    "a recognizable extension."
                ),
                "allowed_extensions": sorted(
                    ALLOWED_EXTENSIONS
                ),
            },
        )

    return extension


def validate_audio_metadata(
    file: UploadFile,
    extension: str,
) -> None:
    """
    Validate the filename extension and MIME type.

    The extension is treated as the main validation signal because
    browsers may report inconsistent MIME types for valid recordings.
    """

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_audio_format",
                "message": (
                    f"Unsupported file extension: "
                    f"{extension}"
                ),
                "allowed_extensions": sorted(
                    ALLOWED_EXTENSIONS
                ),
            },
        )

    normalized_content_type = (
        file.content_type.split(";")[0].strip().lower()
        if file.content_type
        else None
    )

    # Do not reject a valid extension just because the browser
    # supplied no MIME type.
    if not normalized_content_type:
        return

    if normalized_content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_content_type",
                "message": (
                    f"Unsupported content type: "
                    f"{file.content_type}"
                ),
                "allowed_content_types": sorted(
                    ALLOWED_CONTENT_TYPES
                ),
            },
        )


def validate_file_size(file_size: int) -> None:
    """
    Reject empty files and files larger than the configured limit.
    """

    if file_size <= 0:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "empty_audio_file",
                "message": (
                    "The uploaded audio file is empty."
                ),
            },
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "audio_file_too_large",
                "message": (
                    "The uploaded audio file exceeds "
                    "the allowed size limit."
                ),
                "max_file_size_bytes": (
                    MAX_FILE_SIZE_BYTES
                ),
            },
        )


def validate_language_hint(
    language_hint: str,
) -> None:
    """
    Validate the requested language mode.
    """

    if language_hint not in SUPPORTED_LANGUAGE_HINTS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unsupported_language_hint",
                "message": (
                    f"Unsupported language hint: "
                    f"{language_hint}"
                ),
                "supported_language_hints": sorted(
                    SUPPORTED_LANGUAGE_HINTS
                ),
            },
        )