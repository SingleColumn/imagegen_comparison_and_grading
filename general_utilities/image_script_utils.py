import base64
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


def read_prompt_from_file(filepath: str = "prompts.txt", *, strip_wrapping_quotes: bool = False) -> str:
    """Read the prompt from a text file.

    Args:
        filepath: Path to prompt text file.
        strip_wrapping_quotes: If True, remove a single pair of wrapping "..." or '...'.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        if not prompt:
            raise ValueError("Prompt file is empty")

        if strip_wrapping_quotes and len(prompt) >= 2:
            if prompt.startswith('"') and prompt.endswith('"'):
                prompt = prompt[1:-1]
            elif prompt.startswith("'") and prompt.endswith("'"):
                prompt = prompt[1:-1]

        return prompt
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{filepath}' not found")
    except Exception as e:
        raise Exception(f"Error reading prompt file: {e}")


def get_api_key(
    env_var_name: str,
    *,
    missing_message: Optional[str] = None,
) -> str:
    """Get an API key from an environment variable."""
    api_key = os.getenv(env_var_name)
    if not api_key:
        if missing_message:
            raise ValueError(missing_message)
        raise ValueError(f"{env_var_name} environment variable not found. Please set it before running the script.")
    return api_key


def create_output_directory(directory: str = "generated_images") -> str:
    """Create the output directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def sanitize_for_filename(value: str, *, replace_slash: bool = False) -> str:
    """Basic filename-safe sanitizer matching current script behavior."""
    safe = value.replace("-", "_")
    if replace_slash:
        safe = safe.replace("/", "_")
    return safe


def timestamp_string(*, include_seconds: bool, include_milliseconds: bool, suffix_z: bool) -> str:
    """Create a timestamp string compatible with current script conventions."""
    if include_seconds:
        base = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        if include_milliseconds:
            base = base[:-3]
        else:
            base = base[:-6]
    else:
        base = datetime.now().strftime("%Y-%m-%dT%H-%M")

    return base + ("Z" if suffix_z else "")


def generate_timestamp_filename(
    prefix: str,
    main_label: str,
    *,
    extra_label: Optional[str] = None,
    ext: str = ".png",
    replace_slash_in_main_label: bool = False,
    include_seconds: bool = True,
    include_milliseconds: bool = True,
    suffix_z: bool = False,
) -> str:
    """Generate a timestamp filename for an image run."""
    main_safe = sanitize_for_filename(main_label, replace_slash=replace_slash_in_main_label)
    ts = timestamp_string(
        include_seconds=include_seconds,
        include_milliseconds=include_milliseconds,
        suffix_z=suffix_z,
    )

    parts = [p for p in [prefix, main_safe, sanitize_for_filename(extra_label) if extra_label else None, ts] if p]
    return "_".join(parts) + ext


def generate_run_name(
    prefix: str,
    main_label: str,
    *,
    extra_label: Optional[str] = None,
    replace_slash_in_main_label: bool = False,
    include_seconds: bool = True,
    include_milliseconds: bool = True,
    suffix_z: bool = False,
) -> str:
    """Generate a run name (same as filename but without extension)."""
    filename = generate_timestamp_filename(
        prefix=prefix,
        main_label=main_label,
        extra_label=extra_label,
        ext="",
        replace_slash_in_main_label=replace_slash_in_main_label,
        include_seconds=include_seconds,
        include_milliseconds=include_milliseconds,
        suffix_z=suffix_z,
    )
    return filename


@dataclass(frozen=True)
class ImagePayload:
    """Represents an image payload that is either a URL or base64 content."""
    data: str
    is_base64: bool = False


def save_image(payload: str | ImagePayload, filepath: str, *, is_base64: bool = False) -> None:
    """Save an image to disk from a URL or base64 payload.

    Backwards compatible with existing scripts:
      - FAL script passes a URL string.
      - OpenAI script passes a string + is_base64 flag.
    """
    if payload is None:
        raise Exception("Cannot save image: payload is None")

    if isinstance(payload, ImagePayload):
        data = payload.data
        base64_flag = payload.is_base64
    else:
        data = payload
        base64_flag = is_base64

    if not isinstance(data, str):
        raise Exception(f"Cannot save image: payload must be a string, got {type(data).__name__}")

    try:
        if base64_flag:
            image_bytes = base64.b64decode(data)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            print(f"Image saved from base64 data to: {filepath}")
        else:
            urllib.request.urlretrieve(data, filepath)
            print(f"Image downloaded and saved to: {filepath}")
    except Exception as e:
        raise Exception(f"Error saving image: {e}")


