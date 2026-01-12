#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRY DIFFERENT MODELS FOR A GIVEN PROMPT AND SAVE THE IMAGES TO A FOLDER.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from general_utilities.image_generation_utils import (
    call_fal_and_save,
    call_openai_and_save,
    prompt_choice_index,
    read_all_api_keys_from_env,
    safe_filename,
)
from general_utilities.image_script_utils import (
    create_output_directory,
    timestamp_string,
)
from image_evaluator import image_evaluator
from model_config import MODELS


def prompt_prompt_selection(prompts: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Prompt user to select one or more prompts from the prompts list.
    Returns a list of (prompt_name, prompt_text, prompt_dict) tuples.
    """
    if not prompts:
        raise ValueError("No prompts available.")
    
    # Build prompt options for display
    prompt_options: List[str] = []
    for p in prompts:
        name = p.get("prompt_name", f"prompt_{len(prompt_options) + 1}")
        text_preview = p.get("prompt", "")[:60]
        if len(p.get("prompt", "")) > 60:
            text_preview += "..."
        prompt_options.append(f"{name} — {text_preview}")
    
    print("\nSelect prompts to use (enter numbers separated by commas, e.g., 1,3,5, or 'all' for all prompts):")
    print("-" * 70)
    for i, opt in enumerate(prompt_options, start=1):
        print(f"{i}. {opt}")
    print("-" * 70)
    
    while True:
        try:
            raw_input = input(f"Enter prompt numbers (1-{len(prompts)}, comma-separated) or 'all': ").strip().lower()
            if not raw_input:
                print("Please enter at least one prompt number or 'all'.")
                continue
            
            # Check for "all" option
            if raw_input in ("all", "*"):
                selected_indices = list(range(len(prompts)))
                break
            
            indices_str = [s.strip() for s in raw_input.split(",")]
            selected_indices = []
            for idx_str in indices_str:
                try:
                    idx = int(idx_str) - 1  # Convert to 0-based index
                    if 1 <= int(idx_str) <= len(prompts):
                        selected_indices.append(idx)
                    else:
                        print(f"Warning: {idx_str} is out of range. Ignoring.")
                except ValueError:
                    print(f"Warning: '{idx_str}' is not a valid number. Ignoring.")
            
            if not selected_indices:
                print("Please enter at least one valid prompt number or 'all'.")
                continue
            
            break
        except EOFError:
            print("\nNo interactive input detected. Using first prompt as default.")
            selected_indices = [0]
            break
    
    # Get selected prompts, removing duplicates
    selected_prompts: List[Tuple[str, str, Dict[str, Any]]] = []
    seen_indices = set()
    for idx in selected_indices:
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        
        prompt_entry = prompts[idx]
        prompt_name = prompt_entry.get("prompt_name", f"prompt_{idx + 1}")
        prompt_text = prompt_entry.get("prompt", "")
        
        if not prompt_text:
            print(f"Warning: Prompt '{prompt_name}' has no prompt text. Skipping.")
            continue
        
        selected_prompts.append((prompt_name, prompt_text, prompt_entry))
    
    if not selected_prompts:
        raise ValueError("No valid prompts selected.")
    
    return selected_prompts


def prompt_model_selection() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Prompt user to select one or more models from MODELS.
    Returns a list of (model_name, model_cfg) tuples.
    """
    if not MODELS:
        raise ValueError("No models available in configuration.")
    
    # Build model options for display
    model_options: List[str] = []
    for m in MODELS:
        display = m.get("display_name") or m.get("model_name") or m.get("id") or "unknown"
        desc = m.get("description")
        provider = m.get("provider", "unknown")
        model_options.append(f"{display} ({provider})" + (f" — {desc}" if desc else ""))
    
    print("\nSelect models to use (enter numbers separated by commas, e.g., 1,3,5, or 'all' for all models):")
    print("-" * 70)
    for i, opt in enumerate(model_options, start=1):
        print(f"{i}. {opt}")
    print("-" * 70)
    
    while True:
        try:
            raw_input = input(f"Enter model numbers (1-{len(MODELS)}, comma-separated) or 'all': ").strip().lower()
            if not raw_input:
                print("Please enter at least one model number or 'all'.")
                continue
            
            # Check for "all" option
            if raw_input in ("all", "*"):
                selected_indices = list(range(len(MODELS)))
                break
            
            indices_str = [s.strip() for s in raw_input.split(",")]
            selected_indices = []
            for idx_str in indices_str:
                try:
                    idx = int(idx_str) - 1  # Convert to 0-based index
                    if 1 <= int(idx_str) <= len(MODELS):
                        selected_indices.append(idx)
                    else:
                        print(f"Warning: {idx_str} is out of range. Ignoring.")
                except ValueError:
                    print(f"Warning: '{idx_str}' is not a valid number. Ignoring.")
            
            if not selected_indices:
                print("Please enter at least one valid model number or 'all'.")
                continue
            
            break
        except EOFError:
            print("\nNo interactive input detected. Using first model as default.")
            selected_indices = [0]
            break
    
    # Get model configs, removing duplicates
    model_configs: List[Tuple[str, Dict[str, Any]]] = []
    seen_indices = set()
    for idx in selected_indices:
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        
        model_cfg = MODELS[idx]
        model_name = model_cfg.get("model_name") or model_cfg.get("id") or f"model_{idx}"
        model_configs.append((model_name, model_cfg))
    
    if not model_configs:
        raise ValueError("No valid models selected.")
    
    return model_configs


def main() -> int:
    """Main function to orchestrate the prompt testing process across multiple models."""
    try:
        # Read prompt test set
        print("Reading prompt test set...")
        prompt_test_set_path = Path("PROMPTS/prompt_test_set.json")
        if not prompt_test_set_path.exists():
            raise FileNotFoundError(f"Prompt test set file not found: {prompt_test_set_path}")
        
        with open(prompt_test_set_path, "r", encoding="utf-8") as f:
            prompt_test_set_data = json.load(f)
        
        # Handle both list format and nested format with "prompts" key
        if isinstance(prompt_test_set_data, list):
            prompts = prompt_test_set_data
        elif isinstance(prompt_test_set_data, dict) and "prompts" in prompt_test_set_data:
            prompts = prompt_test_set_data["prompts"]
        else:
            raise ValueError("Prompt test set file must contain a list or a dict with 'prompts' key")
        
        print(f"Found {len(prompts)} prompts in test set.\n")

        # Let user select one or more prompts
        selected_prompts = prompt_prompt_selection(prompts)
        print(f"\nSelected {len(selected_prompts)} prompt(s):")
        for prompt_name, prompt_text, _ in selected_prompts:
            print(f"  - {prompt_name}: {prompt_text[:60]}{'...' if len(prompt_text) > 60 else ''}")
        print()

        # Select models from MODELS
        model_configs = prompt_model_selection()
        print(f"\nSelected {len(model_configs)} model(s) to test.\n")

        # Read API keys for all models
        all_model_configs = [cfg for _, cfg in model_configs]
        api_keys = read_all_api_keys_from_env(all_model_configs)

        # Create base output directory with timestamp
        timestamp = timestamp_string(include_seconds=True, include_milliseconds=True, suffix_z=True)
        if len(selected_prompts) == 1:
            # Single prompt: use old format
            prompt_name, _, _ = selected_prompts[0]
            prompt_part = safe_filename(prompt_name)
            run_folder_name = f"{prompt_part}_{timestamp}"
        else:
            # Multiple prompts: use multi_prompt format
            run_folder_name = f"multi_prompt_test_{timestamp}"
        
        base_output_dir = create_output_directory("generated_images")
        output_dir = Path(base_output_dir) / run_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}\n")

        # Initialize report list
        report_data: List[Dict[str, Any]] = []
        
        # Track total costs
        total_generation_cost = 0.0
        total_grading_cost = 0.0

        # Pre-select quality for each OpenAI model (once per model, not per prompt)
        model_quality_map: Dict[str, Optional[str]] = {}
        print("Selecting quality settings for OpenAI models (this will apply to all prompts)...\n")
        for model_name, model_cfg in model_configs:
            provider = (model_cfg.get("provider") or "").lower().strip()
            if provider == "openai":
                supports_quality = bool(model_cfg.get("request", {}).get("supports", {}).get("quality", False))
                defaults = model_cfg.get("defaults", {}) or {}
                if supports_quality:
                    available_qualities = list(model_cfg.get("available_qualities") or [])
                    if available_qualities:
                        # If multiple quality options, prompt user to select
                        if len(available_qualities) > 1:
                            default_quality = defaults.get("quality")
                            default_q_idx = available_qualities.index(default_quality) if default_quality in available_qualities else 0
                            q_idx = prompt_choice_index(
                                f"Available qualities for {model_name}:",
                                available_qualities,
                                default_index=default_q_idx
                            )
                            model_quality_map[model_name] = available_qualities[q_idx]
                        else:
                            # Only one option, use it automatically
                            model_quality_map[model_name] = available_qualities[0]
                    else:
                        # No available qualities list, try to use default
                        model_quality_map[model_name] = defaults.get("quality")
                else:
                    model_quality_map[model_name] = None
            else:
                model_quality_map[model_name] = None
        print()

        # Ask user if they want to grade the image generations
        enable_grading = False
        try:
            grade_input = input("Do you want to grade the image generations? (y/n): ").strip().lower()
            if grade_input in ("y", "yes", "1", "true"):
                enable_grading = True
                print("Grading enabled. Images will be evaluated after generation.\n")
            else:
                print("Grading disabled. Images will not be evaluated.\n")
        except EOFError:
            print("\nNo interactive input detected. Grading disabled by default.\n")

        # If grading is enabled, extract grading criteria for selected prompts
        grading_criteria_map: Dict[str, Dict[str, Any]] = {}
        if enable_grading:
            for prompt_name, _, prompt_entry in selected_prompts:
                grading_criteria = prompt_entry.get("grading_criteria", {})
                if grading_criteria:
                    grading_criteria_map[prompt_name] = grading_criteria
                else:
                    print(f"Warning: No grading_criteria found for prompt '{prompt_name}'. It will be skipped during grading.")

        # Iterate over prompts and models
        total_tasks = len(selected_prompts) * len(model_configs)
        task_num = 0
        
        print(f"Starting generation for {len(selected_prompts)} prompt(s) × {len(model_configs)} model(s) = {total_tasks} total generation(s)...\n")
        print("=" * 70)

        for prompt_idx, (selected_prompt_name, selected_prompt_text, _) in enumerate(selected_prompts, start=1):
            print(f"\n[PROMPT {prompt_idx}/{len(selected_prompts)}] {selected_prompt_name}")
            print(f"Prompt text: {selected_prompt_text[:100]}{'...' if len(selected_prompt_text) > 100 else ''}")
            print("-" * 70)
            
            # For multiple prompts, create a subdirectory for each prompt
            if len(selected_prompts) > 1:
                prompt_output_dir = output_dir / safe_filename(selected_prompt_name)
                prompt_output_dir.mkdir(parents=True, exist_ok=True)
            else:
                prompt_output_dir = output_dir

            # Iterate over models
            for model_idx, (model_name, model_cfg) in enumerate(model_configs, start=1):
                task_num += 1
                print(f"\n[{task_num}/{total_tasks}] Prompt: {selected_prompt_name} | Model: {model_name}")
                print(f"Provider: {model_cfg.get('provider', 'unknown')}")

                try:
                    provider = (model_cfg.get("provider") or "").lower().strip()
                    
                    # Get quality for OpenAI models (already selected above)
                    selected_quality = model_quality_map.get(model_name)

                    # Generate image
                    if provider == "openai":
                        out_path, gen_time, cost = call_openai_and_save(
                            model_cfg, selected_prompt_text, selected_quality, prompt_output_dir,
                            prompt_name=selected_prompt_name, api_keys=api_keys
                        )
                    elif provider == "fal":
                        out_path, gen_time, cost = call_fal_and_save(
                            model_cfg, selected_prompt_text, prompt_output_dir,
                            prompt_name=selected_prompt_name, api_keys=api_keys
                        )
                    else:
                        raise ValueError(f"Unknown provider: {provider!r}")

                    # Evaluate image if grading is enabled
                    grade_result = None
                    grading_cost = None
                    if enable_grading:
                        grading_criteria = grading_criteria_map.get(selected_prompt_name)
                        if grading_criteria:
                            try:
                                print("  Evaluating image...")
                                grade_result, grading_cost = image_evaluator(out_path, grading_criteria_obj=grading_criteria)
                                print(f"  Grade: {grade_result[:100]}{'...' if len(grade_result) > 100 else ''}")
                                print(f"  Grading cost: ${grading_cost:.6f}")
                                total_grading_cost += grading_cost
                            except Exception as eval_error:
                                print(f"  ✗ Error evaluating image: {eval_error}")
                                grade_result = f"Error: {str(eval_error)}"
                        else:
                            print(f"  ⚠ No grading criteria available for prompt '{selected_prompt_name}'. Skipping evaluation.")

                    # Track generation cost
                    if cost is not None:
                        total_generation_cost += cost
                    
                    # Add to report
                    report_entry = {
                        "prompt_name": selected_prompt_name,
                        "model_name": model_name,
                        "generation_time": round(gen_time, 2),
                        "cost": round(cost, 4) if cost is not None else None,
                    }
                    if enable_grading and grade_result is not None:
                        report_entry["grade"] = grade_result
                    if enable_grading and grading_cost is not None:
                        report_entry["grading_cost"] = round(grading_cost, 6)
                    report_data.append(report_entry)

                    print(f"✓ Saved: {out_path.name}")
                    print(f"  Time: {gen_time:.2f}s, Cost: ${cost:.4f}" if cost else f"  Time: {gen_time:.2f}s")

                except Exception as e:
                    print(f"✗ Error generating image with '{model_name}' for prompt '{selected_prompt_name}': {e}")
                    # Still add entry to report with error info
                    report_entry = {
                        "prompt_name": selected_prompt_name,
                        "model_name": model_name,
                        "generation_time": None,
                        "cost": None,
                        "error": str(e),
                    }
                    report_data.append(report_entry)
                    continue

        # Save report
        report_path = output_dir / "image_generation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Report saved: {report_path.name}")

        # Calculate and display total costs
        total_cost = total_generation_cost + total_grading_cost
        print("\n" + "=" * 70)
        print(f"\nCompleted! Generated images for {len(selected_prompts)} prompt(s) × {len(model_configs)} model(s).")
        print(f"Images saved to: {output_dir}")
        print(f"Report saved to: {report_path}")
        print("\n" + "=" * 70)
        print("COST SUMMARY:")
        print("-" * 70)
        print(f"Total image generation cost: ${total_generation_cost:.6f}")
        if enable_grading:
            print(f"Total grading cost:         ${total_grading_cost:.6f}")
            print(f"Total cost (generation + grading): ${total_cost:.6f}")
        else:
            print("Grading was not enabled.")
        print("=" * 70)

        return 0

    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

