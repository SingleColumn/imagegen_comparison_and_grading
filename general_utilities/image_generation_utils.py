#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared utilities for image generation scripts.
Contains common functions used across multiple image generation scripts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from general_utilities.image_script_utils import (
    get_api_key,
    save_image,
    sanitize_for_filename,
    timestamp_string,
)


def prompt_choice_index(
    title: str,
    options: Sequence[str],
    default_index: Optional[int] = None,
) -> int:
    """Prompt user to select from a list of options by index."""
    if not options:
        raise ValueError("No options provided.")

    print("\n" + title)
    print("-" * 70)
    for i, opt in enumerate(options, start=1):
        print(f"{i}. {opt}")
    print("-" * 70)

    while True:
        default_hint = ""
        if default_index is not None:
            default_hint = f" or press Enter for default ({default_index + 1})"
        try:
            raw = input(f"Select an option (1-{len(options)}){default_hint}: ").strip()
        except EOFError:
            # Non-interactive execution (e.g., piped/automated). Fall back to default if present.
            if default_index is not None:
                print("\nNo interactive input detected; using default.")
                return default_index
            raise

        if raw == "" and default_index is not None:
            return default_index
        try:
            n = int(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if 1 <= n <= len(options):
            return n - 1
        print(f"Please enter a number between 1 and {len(options)}.")


def prompt_yes_no(question: str, default: bool = False) -> bool:
    """Prompt user for yes/no answer."""
    default_hint = "Y/n" if default else "y/N"
    default_answer = "yes" if default else "no"
    
    while True:
        try:
            raw = input(f"{question} ({default_hint}): ").strip().lower()
        except EOFError:
            print(f"\nNo interactive input detected; using default: {default_answer}.")
            return default
        
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def safe_filename(s: str) -> str:
    """Enhanced filename sanitizer."""
    safe = sanitize_for_filename(s, replace_slash=True)
    return (
        safe.replace(":", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("·", "")
    )


def extract_by_path(obj: Any, path: List[Union[str, int]]) -> Any:
    """Walks an object/dict/list using a path like ["images", 0, "url"]."""
    cur = obj
    for key in path:
        if cur is None:
            return None
        if isinstance(key, int):
            if isinstance(cur, (list, tuple)) and 0 <= key < len(cur):
                cur = cur[key]
            else:
                return None
        else:
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                cur = getattr(cur, key, None)
    return cur


def calculate_openai_cost(model_cfg: Dict[str, Any], quality: Optional[str], size: str) -> Optional[float]:
    """Calculate cost for OpenAI models using pricing table from model config."""
    pricing_table = model_cfg.get("pricing_usd_per_image")
    if not pricing_table:
        return None
    
    # Use quality or default
    defaults = model_cfg.get("defaults", {}) or {}
    quality_to_use = quality or defaults.get("quality")
    if not quality_to_use:
        return None
    
    quality_pricing = pricing_table.get(quality_to_use)
    if not quality_pricing:
        return None
    
    return quality_pricing.get(size)


def call_openai_and_save(
    model_cfg: Dict[str, Any],
    prompt: str,
    quality: Optional[str],
    out_dir: Path,
    prompt_name: Optional[str] = None,
    api_keys: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Path, float, Optional[float]]:
    """Generate image using OpenAI API and return (path, generation_time, cost)."""
    try:
        import openai  # type: ignore
    except ImportError:
        raise ImportError("Missing dependency `openai`. Install with: `py -m pip install -r requirements.txt`")

    env_name = model_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = (api_keys or {}).get(env_name)
    if not api_key:
        missing_msg = (
            f"Missing environment variable `{env_name}`.\n"
            f"In PowerShell you can set it for this session with:\n"
            f"  $env:{env_name} = 'YOUR_KEY_HERE'\n"
        )
        api_key = get_api_key(env_name, missing_message=missing_msg)
    client = openai.OpenAI(api_key=api_key)

    request = model_cfg["request"]
    if request.get("method") != "images.generate":
        raise ValueError(f"Unsupported OpenAI request method: {request.get('method')}")

    params: Dict[str, Any] = dict(request.get("params", {}))
    params["prompt"] = prompt

    # size support is described in config; default if present
    defaults = model_cfg.get("defaults", {}) or {}
    if request.get("supports", {}).get("size", False):
        size = defaults.get("size") or (model_cfg.get("available_sizes") or [None])[0]
        if not size:
            raise ValueError("OpenAI model requires `size` but none was configured.")
        params["size"] = size
    else:
        size = "1024x1024"  # fallback

    if request.get("supports", {}).get("quality", False) and quality:
        params["quality"] = quality

    t0 = time.time()
    resp = client.images.generate(**params)
    dt = time.time() - t0
    print(f"OpenAI image generation completed in {dt:.2f}s")

    if not getattr(resp, "data", None) or len(resp.data) == 0:
        raise RuntimeError("OpenAI response contained no data.")

    item0 = resp.data[0]
    response_cfg = model_cfg.get("response", {}) or {}
    fields_priority = response_cfg.get("image_fields_priority") or ["url", "b64_json"]

    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    for f in fields_priority:
        v = getattr(item0, f, None)
        if not v:
            continue
        if f == "url":
            image_url = v
            break
        if f == "b64_json":
            image_b64 = v
            break

    if not image_url and not image_b64:
        available = [a for a in dir(item0) if not a.startswith("_")]
        raise RuntimeError(f"Could not find image content in OpenAI response. Available fields: {available}")

    model_part = safe_filename(model_cfg.get("id", model_cfg.get("model_name", "openai")))
    quality_part = safe_filename(quality or defaults.get("quality") or "default")
    timestamp = timestamp_string(include_seconds=True, include_milliseconds=True, suffix_z=True)
    prompt_part = safe_filename(prompt_name) if prompt_name else ""
    parts = [p for p in [model_part, quality_part, prompt_part, timestamp] if p]
    filename = "_".join(parts) + ".png"
    out_path = out_dir / filename
    # Ensure the output directory exists before saving
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if image_b64:
        save_image(image_b64, str(out_path), is_base64=True)
    else:
        assert image_url is not None
        save_image(image_url, str(out_path), is_base64=False)

    # Calculate cost
    cost = calculate_openai_cost(model_cfg, quality, params.get("size", size))

    return out_path, dt, cost


def wait_for_fal_completion(model_path: str, request_id: str, max_wait_time_s: int = 600, poll_interval_s: int = 2) -> None:
    """Wait for FAL request to complete."""
    import fal_client  # type: ignore

    start = time.time()
    last_status_time = start
    status_check_count = 0
    
    while True:
        try:
            status = fal_client.status(model_path, request_id, with_logs=True)
            status_check_count += 1
        except Exception as e:
            raise RuntimeError(f"Error checking FAL status (model_path='{model_path}', request_id='{request_id}'): {e}")

        # fal_client sometimes returns typed objects; sometimes dict-like
        status_type = type(status).__name__
        if "Completed" in status_type:
            elapsed = time.time() - start
            print(f"  Completed in {elapsed:.1f}s (after {status_check_count} status checks)")
            return
        if "Failed" in status_type:
            raise RuntimeError(f"FAL request failed: {status}")

        status_value = None
        if isinstance(status, dict):
            status_value = status.get("status")
        else:
            status_value = getattr(status, "status", None)

        if status_value == "COMPLETED":
            elapsed = time.time() - start
            print(f"  Completed in {elapsed:.1f}s (after {status_check_count} status checks)")
            return
        if status_value == "FAILED":
            raise RuntimeError(f"FAL request failed: {status}")

        elapsed = time.time() - start
        if elapsed > max_wait_time_s:
            raise TimeoutError(f"FAL request timed out after {max_wait_time_s} seconds ({max_wait_time_s/60:.1f} minutes) (request_id={request_id})")
        
        # Print progress every 30 seconds
        if time.time() - last_status_time >= 30:
            remaining = max_wait_time_s - elapsed
            print(f"  Still processing... (elapsed: {elapsed:.0f}s, remaining: ~{remaining:.0f}s)")
            last_status_time = time.time()
        
        time.sleep(poll_interval_s)


def get_fal_pricing(model_path: str, api_key: str) -> Optional[float]:
    """Get pricing information for a FAL model endpoint using the Pricing API.
    
    Returns the unit price per image in USD, or None if unavailable.
    """
    if not REQUESTS_AVAILABLE:
        return None
    
    try:
        url = "https://api.fal.ai/v1/models/pricing"
        headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }
        params = {"endpoint_id": model_path}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])
            if prices and len(prices) > 0:
                unit_price = prices[0].get("unit_price")
                if unit_price is not None:
                    return float(unit_price)
        return None
    except Exception:
        # Silently fail - pricing API is optional
        return None


def get_fal_usage_cost(request_id: str, api_key: str) -> Optional[float]:
    """Get actual cost for a FAL request using the Usage API.
    
    Returns the actual cost in USD, or None if unavailable.
    """
    if not REQUESTS_AVAILABLE:
        return None
    
    try:
        url = "https://api.fal.ai/v1/models/usage"
        headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }
        params = {"request_id": request_id}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # The usage API may return a list of usage records
            if isinstance(data, list) and len(data) > 0:
                usage_record = data[0]
                cost = usage_record.get("cost") or usage_record.get("price")
                if cost is not None:
                    return float(cost)
            elif isinstance(data, dict):
                cost = data.get("cost") or data.get("price")
                if cost is not None:
                    return float(cost)
        return None
    except Exception:
        # Silently fail - usage API is optional
        return None


def call_fal_and_save(
    model_cfg: Dict[str, Any],
    prompt: str,
    out_dir: Path,
    prompt_name: Optional[str] = None,
    api_keys: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Path, float, Optional[float]]:
    """Generate image using FAL API and return (path, generation_time, cost)."""
    try:
        import fal_client  # type: ignore
    except ImportError:
        raise ImportError("Missing dependency `fal-client`. Install with: `py -m pip install -r requirements.txt`")

    env_name = model_cfg.get("api_key_env", "FAL_KEY")
    api_key = (api_keys or {}).get(env_name)
    if not api_key:
        missing_msg = (
            f"Missing environment variable `{env_name}`.\n"
            f"In PowerShell you can set it for this session with:\n"
            f"  $env:{env_name} = 'YOUR_KEY_HERE'\n"
        )
        api_key = get_api_key(env_name, missing_message=missing_msg)
    # fal_client reads from env
    os.environ["FAL_KEY"] = api_key

    request = model_cfg["request"]
    if request.get("method") != "fal_client.submit":
        raise ValueError(f"Unsupported FAL request method: {request.get('method')}")

    model_path = request.get("model_path")
    if not model_path:
        raise ValueError("FAL model is missing `request.model_path`.")

    print(f"  Using model_path: {model_path}")

    # Get pricing information for cost estimation
    estimated_cost = get_fal_pricing(model_path, api_key)

    args_template: Dict[str, Any] = dict(request.get("arguments_template") or {})
    # minimal templating: format any str values with {prompt}
    arguments: Dict[str, Any] = {}
    for k, v in args_template.items():
        if isinstance(v, str):
            arguments[k] = v.format(prompt=prompt)
        else:
            arguments[k] = v

    t0 = time.time()
    try:
        handler = fal_client.submit(model_path, arguments=arguments)
        request_id = handler.request_id
        print(f"  FAL request submitted. request_id={request_id}")
    except Exception as e:
        raise RuntimeError(f"Error submitting FAL request with model_path '{model_path}': {e}")

    # Get timeout from model config if specified, otherwise use default (600s = 10 minutes)
    max_wait_time = model_cfg.get("max_wait_time_s") or 600
    try:
        print(f"  Waiting for completion (timeout: {max_wait_time}s)...")
        wait_for_fal_completion(model_path, request_id, max_wait_time_s=max_wait_time)
    except Exception as e:
        raise RuntimeError(f"Error waiting for FAL completion (model_path='{model_path}', request_id='{request_id}'): {e}")

    # Get the final status (may contain usage/cost information)
    try:
        final_status = fal_client.status(model_path, request_id, with_logs=True)
    except Exception as e:
        raise RuntimeError(f"Error getting FAL status (model_path='{model_path}', request_id='{request_id}'): {e}")
    
    # Get the result
    try:
        result = fal_client.result(model_path, request_id)
    except Exception as e:
        raise RuntimeError(f"Error getting FAL result (model_path='{model_path}', request_id='{request_id}'): {e}")
    
    dt = time.time() - t0
    print(f"  FAL image generation completed in {dt:.2f}s")

    # Extract cost information from status or result (actual cost from API response)
    cost = None
    # Check status for cost/usage information
    if isinstance(final_status, dict):
        cost = final_status.get('cost')
    else:
        cost = getattr(final_status, 'cost', None)
    
    # Check result for cost/usage information if not found in status
    if cost is None:
        if isinstance(result, dict):
            cost = result.get('cost')
        else:
            cost = getattr(result, 'cost', None)
    
    # Try Usage API to get actual cost for this request
    if cost is None:
        cost = get_fal_usage_cost(request_id, api_key)
    
    # Fall back to estimated cost from pricing API if actual cost unavailable
    if cost is None:
        cost = estimated_cost

    response_cfg = model_cfg.get("response", {}) or {}
    url_paths: List[List[Union[str, int]]] = response_cfg.get("image_url_paths_priority") or [
        ["images", 0, "url"],
        ["url"],
    ]

    image_url: Optional[str] = None
    for path in url_paths:
        v = extract_by_path(result, path)
        if isinstance(v, str) and v:
            image_url = v
            break

    if not image_url:
        if isinstance(result, dict):
            raise RuntimeError(f"Could not locate image URL in FAL result. Keys: {list(result.keys())}")
        raise RuntimeError("Could not locate image URL in FAL result.")

    model_part = safe_filename(model_cfg.get("id", model_cfg.get("model_name", "fal")))
    timestamp = timestamp_string(include_seconds=True, include_milliseconds=True, suffix_z=True)
    prompt_part = safe_filename(prompt_name) if prompt_name else ""
    parts = [p for p in [model_part, prompt_part, timestamp] if p]
    filename = "_".join(parts) + ".png"
    out_path = out_dir / filename
    # Ensure the output directory exists before saving
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image_url, str(out_path), is_base64=False)

    return out_path, dt, cost


def read_all_api_keys_from_env(models: Sequence[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Read all API keys referenced by the configured models from environment variables."""
    env_vars: List[str] = []
    for m in models:
        env_name = m.get("api_key_env")
        if isinstance(env_name, str) and env_name.strip():
            env_vars.append(env_name.strip())

    # de-dupe while preserving order
    seen = set()
    unique_env_vars: List[str] = []
    for e in env_vars:
        if e in seen:
            continue
        seen.add(e)
        unique_env_vars.append(e)

    keys: Dict[str, Optional[str]] = {e: os.getenv(e) for e in unique_env_vars}

    if unique_env_vars:
        print("\nAPI key environment variables:")
        print("-" * 70)
        for e in unique_env_vars:
            status = "SET" if keys.get(e) else "MISSING"
            print(f"{e}: {status}")
        print("-" * 70)

    return keys


def read_prompt_test_set(filepath: str) -> List[Dict[str, Any]]:
    """Read prompts from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing prompts
        
    Returns:
        List of prompt dictionaries
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file doesn't contain a valid list or can't be parsed
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            raise ValueError(f"File should contain a list, got {type(prompts).__name__}")
        return prompts
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt test set file '{filepath}' not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON from {filepath}: {e}")

