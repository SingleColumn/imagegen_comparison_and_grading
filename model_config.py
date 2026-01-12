"""
Central configuration for image-generation models.

This module is intended to be imported by your CLI/workflow scripts.
It provides a single list of dictionaries (MODELS) where each entry contains
everything required to:
- present a model to the user,
- validate/select supported options (quality, size, etc),
- dispatch to the correct provider (OpenAI or FAL),
- build the correct API call parameters/payload.

Design notes:
- Prefer a stable `id` for selection + persistence (filenames, logs).
- Use `provider` to route the call in code ("openai" vs "fal").
- Store request-shape details as data (`request`) rather than storing function objects.
  This keeps config JSON-serializable and avoids import-time side effects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# A list of dictionaries, each describing one selectable image generation model.
MODELS: List[Dict[str, Any]] = [
    # -----------------------
    # OpenAI image generation
    # -----------------------
    {
        # Required keys (per your request)
        "model_name": "gpt-image-1",
        "available_qualities": ["low", "medium", "high"],
        # Recommended keys
        "id": "openai:gpt-image-1",
        "provider": "openai",
        "display_name": "gpt-image-1",
        "description": "OpenAI Images API (supports low/medium/high).",
        "api_key_env": "OPENAI_API_KEY",
        "api_documentation_url": "",
        "request": {
            # For OpenAI Python SDK: client.images.generate(**params)
            "method": "images.generate",
            "params": {
                "model": "gpt-image-1",
                "n": 1,
            },
            # Which user-selectable options this model supports
            "supports": {
                "quality": True,
                "size": True,
            },
        },
        "available_sizes": ["1024x1024", "1024x1536", "1536x1024"],
        "defaults": {
            "quality": "medium",
            "size": "1024x1024",
            "n": 1,
        },
        # Optional: embed pricing table so cost estimation is data-driven
        "pricing_usd_per_image": {
            "low": {"1024x1024": 0.011, "1024x1536": 0.016, "1536x1024": 0.016},
            "medium": {"1024x1024": 0.042, "1024x1536": 0.063, "1536x1024": 0.063},
            "high": {"1024x1024": 0.167, "1024x1536": 0.250, "1536x1024": 0.250},
        },
        # Useful when parsing responses: OpenAI may return url or base64 depending on model/account settings.
        "response": {
            "image_fields_priority": ["url", "b64_json"],
        },
    },
    {
        "model_name": "dall-e-3",
        "available_qualities": ["standard", "hd"],
        "id": "openai:dall-e-3",
        "provider": "openai",
        "display_name": "DALL·E 3",
        "description": "OpenAI Images API (standard/hd).",
        "api_key_env": "OPENAI_API_KEY",
        "api_documentation_url": "",
        "request": {
            "method": "images.generate",
            "params": {
                "model": "dall-e-3",
                "n": 1,
            },
            "supports": {
                "quality": True,
                "size": True,
            },
        },
        "available_sizes": ["1024x1024", "1024x1792", "1792x1024"],
        "defaults": {
            "quality": "standard",
            "size": "1024x1024",
            "n": 1,
        },
        "pricing_usd_per_image": {
            "standard": {"1024x1024": 0.040, "1024x1792": 0.080, "1792x1024": 0.080},
            "hd": {"1024x1024": 0.080, "1024x1792": 0.120, "1792x1024": 0.120},
        },
        "response": {
            "image_fields_priority": ["url", "b64_json"],
        },
    },
    {
        "model_name": "dall-e-2",
        "available_qualities": ["standard"],
        "id": "openai:dall-e-2",
        "provider": "openai",
        "display_name": "DALL·E 2",
        "description": "OpenAI Images API (quality fixed to standard).",
        "api_key_env": "OPENAI_API_KEY",
        "api_documentation_url": "",
        "request": {
            "method": "images.generate",
            "params": {
                "model": "dall-e-2",
                "n": 1,
            },
            "supports": {
                "quality": False,  # dall-e-2 does not accept `quality`
                "size": True,
            },
        },
        "available_sizes": ["256x256", "512x512", "1024x1024"],
        "defaults": {
            "quality": "standard",
            "size": "1024x1024",
            "n": 1,
        },
        "pricing_usd_per_image": {
            "standard": {"256x256": 0.016, "512x512": 0.018, "1024x1024": 0.020},
        },
        "response": {
            "image_fields_priority": ["url", "b64_json"],
        },
    },
    # -----------------------
    # FAL.ai image generation
    # -----------------------
    {
        "model_name": "flux-pro/v1.1-ultra",
        "available_qualities": [],  # not used in your current FAL script
        "id": "fal:fal-ai/flux-pro/v1.1-ultra",
        "provider": "fal",
        "display_name": "flux-pro/v1.1-ultra",
        "description": "FAL.ai model path: fal-ai/flux-pro/v1.1-ultra",
        "api_key_env": "FAL_KEY",
        "api_documentation_url": "https://fal.ai/models/fal-ai/flux-pro/v1.1-ultra/api",
        "request": {
            # For fal_client: fal_client.submit(model_path, arguments={...})
            "method": "fal_client.submit",
            "model_path": "fal-ai/flux-pro/v1.1-ultra",
            "arguments_template": {
                "prompt": "{prompt}",
                "aspect_ratio": "1:1",
                "guidance_scale": 5,  # High prompt strength (default)
            },
        },
        "defaults": {},
        "response": {
            # Your FAL script expects result.images[0].url typically.
            "image_url_paths_priority": [["images", 0, "url"], ["images", 0, "image_url"], ["url"], ["image_url"]],
        },
    },
    {
        "model_name": "recraft/v3",
        "available_qualities": [],
        "id": "fal:fal-ai/recraft/v3/text-to-image",
        "provider": "fal",
        "display_name": "recraft/v3",
        "description": "FAL.ai model path: fal-ai/recraft/v3/text-to-image",
        "api_key_env": "FAL_KEY",
        "api_documentation_url": "https://fal.ai/models/fal-ai/recraft/v3/text-to-image/api",
        "request": {
            "method": "fal_client.submit",
            "model_path": "fal-ai/recraft/v3/text-to-image",
            "arguments_template": {
                "prompt": "{prompt}",
                "image_size": "square",
            },
        },
        "defaults": {},
        "response": {
            "image_url_paths_priority": [["images", 0, "url"], ["images", 0, "image_url"], ["url"], ["image_url"]],
        },
    },
    {
        "model_name": "flux-2",
        "available_qualities": [],
        "id": "fal:fal-ai/flux-2",
        "provider": "fal",
        "display_name": "flux-2 – FLUX.2 [dev]",
        "description": "FAL.ai model path: fal-ai/flux-2",
        "api_key_env": "FAL_KEY",
        "api_documentation_url": "https://fal.ai/models/fal-ai/flux-2/api",
        "request": {
            "method": "fal_client.submit",
            "model_path": "fal-ai/flux-2",
            "arguments_template": {
                "prompt": "{prompt}",
                "image_size": "square",
            },
        },
        "defaults": {},
        "response": {
            "image_url_paths_priority": [["images", 0, "url"], ["images", 0, "image_url"], ["url"], ["image_url"]],
        },
    },
    {
        "model_name": "bria/text-to-image/3.2",
        "available_qualities": [],
        "id": "fal:bria/text-to-image/3.2",
        "provider": "fal",
        "display_name": "bria/text-to-image/3.2",
        "description": "FAL.ai model path: bria/text-to-image/3.2",
        "api_key_env": "FAL_KEY",
        "api_documentation_url": "https://fal.ai/models/bria/text-to-image/3.2/api",
        "request": {
            "method": "fal_client.submit",
            "model_path": "bria/text-to-image/3.2",
            "arguments_template": {
                "prompt": "{prompt}",
                "aspect_ratio": "1:1",
            },
        },
        "defaults": {},
        "response": {
            "image_url_paths_priority": [["image", "url"], ["url"], ["image_url"]],
        },
    },
    {
        "model_name": "imagen4/preview/fast",
        "available_qualities": [],
        "id": "fal:fal-ai/imagen4/preview/fast",
        "provider": "fal",
        "display_name": "imagen4/preview/fast",
        "description": "FAL.ai model path: fal-ai/imagen4/preview/fast",
        "api_key_env": "FAL_KEY",
        "api_documentation_url": "https://fal.ai/models/fal-ai/imagen4/preview/fast/api",
        "request": {
            "method": "fal_client.submit",
            "model_path": "fal-ai/imagen4/preview/fast",
            "arguments_template": {
                "prompt": "{prompt}",
                "image_size": "square",
            },
        },
        "defaults": {},
        "response": {
            "image_url_paths_priority": [["images", 0, "url"], ["images", 0, "image_url"], ["url"], ["image_url"]],
        },
    },
    {
        "model_name": "HiDream-I1 full",
        "available_qualities": [],
        "id": "fal:fal-ai/hidream-i1-full",
        "provider": "fal",
        "display_name": "HiDream-I1 full",
        "description": "FAL.ai model path: fal-ai/hidream-i1-full",
        "api_key_env": "FAL_KEY",
        "api_documentation_url": "https://fal.ai/models/fal-ai/hidream-i1-full/api",
        "request": {
            "method": "fal_client.submit",
            "model_path": "fal-ai/hidream-i1-full",
            "arguments_template": {
                "prompt": "{prompt}",
                "image_size": "square",
            },
        },
        "defaults": {},
        "response": {
            "image_url_paths_priority": [["images", 0, "url"], ["images", 0, "image_url"], ["url"], ["image_url"]],
        },
    },
    {
        "model_name": "HiDream-I1 fast",
        "available_qualities": [],
        "id": "fal:fal-ai/HiDream-I1-fast",
        "provider": "fal",
        "display_name": "HiDream-I1 fast – Faster variant (16 steps)",
        "description": "FAL.ai model path: fal-ai/HiDream-I1-fast",
        "api_key_env": "FAL_KEY",
        "api_documentation_url": "https://fal.ai/models/fal-ai/hidream-i1-fast",
        "request": {
            "method": "fal_client.submit",
            "model_path": "fal-ai/HiDream-I1-fast",
            "arguments_template": {
                "prompt": "{prompt}",
                "image_size": "square",
            },
        },
        "defaults": {},
        "response": {
            "image_url_paths_priority": [["images", 0, "url"], ["images", 0, "image_url"], ["url"], ["image_url"]],
        },
    },
]


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Return the model dict for a given stable `id`."""
    for m in MODELS:
        if m.get("id") == model_id:
            return m
    return None


def list_model_choices() -> List[Dict[str, str]]:
    """
    Convenience for building a selection UI.
    Returns [{"id": "...", "display": "..."}]
    """
    return [{"id": m["id"], "display": m.get("display_name", m["model_name"])} for m in MODELS]


