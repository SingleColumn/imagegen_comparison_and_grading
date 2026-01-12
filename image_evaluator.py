#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
image_evaluator.py

Function to evaluate images using GPT-4o vision API based on grading criteria
from PROMPTS/prompt_test_set_updated.json.
"""

import base64
import json
import os
from pathlib import Path
from typing import Optional

try:
    import openai
except ImportError:
    raise ImportError("Missing dependency `openai`. Install with: `py -m pip install -r requirements.txt`")

from general_utilities.image_script_utils import get_api_key


def _format_grading_criteria(criteria_obj: dict) -> str:
    """
    Format the grading criteria object into a readable string for GPT-4o.
    
    Args:
        criteria_obj: Dictionary containing grading criteria structure
        
    Returns:
        Formatted string with all grading criteria information
    """
    lines = []
    
    # Evaluation instructions
    if "evaluation_instructions" in criteria_obj:
        lines.append("EVALUATION INSTRUCTIONS:")
        lines.append(criteria_obj["evaluation_instructions"])
        lines.append("")
    
    # Hard fail conditions
    if "hard_fail_conditions" in criteria_obj:
        lines.append("HARD FAIL CONDITIONS (if any are present, apply score cap):")
        for condition in criteria_obj["hard_fail_conditions"]:
            lines.append(f"  - {condition}")
        lines.append("")
    
    # Required checks
    if "required_checks" in criteria_obj:
        lines.append("REQUIRED CHECKS:")
        for check in criteria_obj["required_checks"]:
            lines.append(f"  - {check}")
        lines.append("")
    
    # Failure mode handling
    if "failure_mode_handling" in criteria_obj:
        failure_handling = criteria_obj["failure_mode_handling"]
        if "instructions" in failure_handling:
            lines.append("FAILURE MODE HANDLING:")
            lines.append(failure_handling["instructions"])
            lines.append("")
        
        if "severity_levels" in failure_handling:
            severity_levels = failure_handling["severity_levels"]
            for severity, level_data in severity_levels.items():
                score_cap = level_data.get("score_cap", "N/A")
                modes = level_data.get("modes", [])
                if modes:
                    lines.append(f"{severity.upper()} FAILURE MODES (score cap: {score_cap}):")
                    for mode in modes:
                        lines.append(f"  - {mode}")
                    lines.append("")
    
    # Scoring rubric
    if "scoring_rubric" in criteria_obj:
        lines.append("SCORING RUBRIC:")
        rubric = criteria_obj["scoring_rubric"]
        # Sort by score (descending) for better readability
        for score in sorted(rubric.keys(), key=lambda x: int(x) if x.isdigit() else 0, reverse=True):
            lines.append(f"  Score {score}: {rubric[score]}")
        lines.append("")
    
    return "\n".join(lines)


def _calculate_gpt4o_cost(usage) -> float:
    """
    Calculate the cost for a GPT-4o API call based on token usage.
    
    Args:
        usage: Usage object from OpenAI API response (has prompt_tokens, completion_tokens, total_tokens)
        
    Returns:
        Cost in USD
    """
    # GPT-4o pricing (as of 2024):
    # Input: $2.50 per 1M tokens
    # Output: $10.00 per 1M tokens
    INPUT_COST_PER_MILLION = 2.50
    OUTPUT_COST_PER_MILLION = 10.00
    
    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
    completion_tokens = getattr(usage, 'completion_tokens', 0)
    
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    
    return input_cost + output_cost


def image_evaluator(image_path: str | Path, grading_criteria_obj: Optional[dict] = None) -> tuple[str, float]:
    """
    Evaluate an image using GPT-4o vision API based on grading criteria.
    
    Args:
        image_path: Path to the image file to evaluate
        grading_criteria_obj: Optional grading criteria dictionary. If not provided,
                             will be read from prompt_test_set_updated.json based on image filename.
        
    Returns:
        Tuple of (grade/response from GPT-4o, cost in USD)
        
    Raises:
        FileNotFoundError: If image file or prompt_test_set_updated.json not found
        ValueError: If no matching prompt_name found in the image filename or no grading criteria
        Exception: For API errors or other issues
    """
    # Convert to Path object for easier handling
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # If grading criteria not provided, read from JSON file
    if grading_criteria_obj is None:
        # Read prompt_test_set_updated.json
        prompt_test_set_path = Path("PROMPTS/prompt_test_set_updated.json")
        if not prompt_test_set_path.exists():
            raise FileNotFoundError(f"prompt_test_set_updated.json not found at: {prompt_test_set_path.absolute()}")
        
        with open(prompt_test_set_path, "r", encoding="utf-8") as f:
            prompt_test_set_data = json.load(f)
        
        # Access the prompts array
        prompt_test_set = prompt_test_set_data.get("prompts", [])
        if not prompt_test_set:
            raise ValueError("No prompts found in prompt_test_set_updated.json")
        
        # Extract filename without extension
        image_filename = image_path.stem  # Gets filename without extension
        
        # Find matching item in prompt_test_set
        # Prefer longer matches to avoid partial substring matches
        matching_item = None
        best_match_length = 0
        for item in prompt_test_set:
            prompt_name = item.get("prompt_name", "")
            if prompt_name and prompt_name in image_filename:
                # Prefer longer matches (more specific)
                if len(prompt_name) > best_match_length:
                    matching_item = item
                    best_match_length = len(prompt_name)
        
        if not matching_item:
            raise ValueError(
                f"No matching prompt_name found in image filename '{image_filename}'. "
                f"Available prompt_names: {[item.get('prompt_name') for item in prompt_test_set]}"
            )
        
        # Get grading criteria (now an object, not a string)
        grading_criteria_obj = matching_item.get("grading_criteria", {})
        if not grading_criteria_obj:
            raise ValueError(f"No grading_criteria found for prompt_name: {matching_item.get('prompt_name')}")
    
    # Format grading criteria object into a readable string
    grading_criteria = _format_grading_criteria(grading_criteria_obj)
    
    # Get OpenAI API key from environment variables
    env_name = "OPENAI_API_KEY"
    missing_msg = (
        f"Missing environment variable `{env_name}`.\n"
        f"In PowerShell you can set it for this session with:\n"
        f"  $env:{env_name} = 'YOUR_KEY_HERE'\n"
    )
    # get_api_key reads from os.getenv() to get the value from environment variables
    api_key = get_api_key(env_name, missing_message=missing_msg)
    client = openai.OpenAI(api_key=api_key)
    
    # Read and encode image as base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    # Check image size (OpenAI has a 20MB limit for base64 encoded images)
    # Base64 encoding increases size by ~33%, so we check raw size
    MAX_IMAGE_SIZE_MB = 15  # Leave some margin for base64 overhead
    image_size_mb = len(image_data) / (1024 * 1024)
    
    if image_size_mb > MAX_IMAGE_SIZE_MB:
        raise ValueError(
            f"Image file is too large ({image_size_mb:.2f} MB). "
            f"Maximum size is {MAX_IMAGE_SIZE_MB} MB. "
            f"Please resize the image before processing."
        )
    
    # Validate image can be read (basic check)
    try:
        import io
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            # PIL not available, skip validation
            Image = None
        
        if Image is not None:
            img = Image.open(io.BytesIO(image_data))
            img.verify()  # Verify it's a valid image
    except Exception as e:
        raise ValueError(f"Image file appears to be corrupted or invalid: {e}")
    
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    
    # Determine image MIME type from extension
    image_ext = image_path.suffix.lower()
    mime_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_type_map.get(image_ext, "image/png")
    
    # Construct the prompt for GPT-4o

    HARD_FAIL_RULE = """
HARD FAIL RULE (GLOBAL):
HARD FAIL CONDITIONS represent explicit, binary violations of core prompt constraints.

If ANY HARD FAIL condition listed in the grading criteria is met,
you MUST cap the final score at 2 or lower,
regardless of how many other criteria are satisfied.
"""

    ABSENCE_RULE = """
REQUIRED CHECK & ABSENCE RULE (GLOBAL):
REQUIRED CHECKS enumerate observable elements, relations, or properties that the prompt explicitly demands.

For EACH REQUIRED CHECK, you must explicitly verify that the element is:
- visibly present, AND
- directly evaluable from the image.

If a REQUIRED CHECK is missing, not visible, or cannot be evaluated due to absence or ambiguity:

1. Determine whether this failure corresponds to a listed FAILURE MODE.
2. If so, apply that FAILURE MODE's severity and score cap.
3. If no matching FAILURE MODE exists, treat the failure as at least a MODERATE failure
   and cap the score accordingly.
"""

    FUNCTIONAL_VERIFICATION_RULE = """
FUNCTIONAL VERIFICATION RULE (GLOBAL):
Some REQUIRED CHECKS specify functional or causal behavior
(e.g. refraction, reflection, interaction, alignment, correspondence, containment).

For any such check, you MUST verify direct visual evidence that the behavior is actually occurring.

Plausibility, realism, stylistic resemblance, or visual suggestion is NOT sufficient.
If the image merely appears consistent with the behavior but does not show clear visual evidence of it,
the REQUIRED CHECK is NOT SATISFIED.
"""

    EVIDENCE_STANDARD_RULE = """
EVIDENCE STANDARD (GLOBAL):
When evaluating functional or relational requirements, ask yourself:

\"Can I point to a specific region of the image that demonstrates this behavior,
and would the behavior still be evident if the surrounding context were removed?\"

If the answer is no, the requirement is not satisfied.
"""

    SEVERITY_AND_SCORE_MEANINGS = """
SEVERITY LEVELS (GLOBAL):
- SEVERE failure: core structural, logical, or identity breakdown → score cap = 2
- MODERATE failure: partial degradation under stress → score cap = 4

SCORE MEANINGS (GLOBAL):
Score 5: Complete success under all specified constraints; no meaningful degradation.
Score 4: Minor degradation that does not violate core constraints.
Score 3: Partial success with ambiguity or weakened constraint satisfaction.
Score 2: Major breakdown of required structure, logic, or identity.
Score 1: Failure of the core task or collapse of required elements.
"""

    evaluation_prompt = f"""
You are an image evaluation expert. Analyze the provided image carefully and skeptically.

Your task is to apply the grading criteria exactly as specified, based solely on observable visual evidence.

{HARD_FAIL_RULE}

{ABSENCE_RULE}

{FUNCTIONAL_VERIFICATION_RULE}

{EVIDENCE_STANDARD_RULE}

{SEVERITY_AND_SCORE_MEANINGS}

GRADING CRITERIA (PROMPT-SPECIFIC):
{grading_criteria}

EVALUATION PROCEDURE (YOU MUST FOLLOW THIS ORDER):
1. Check HARD FAIL CONDITIONS. If any are met, apply the hard-fail score cap.
2. Verify EACH REQUIRED CHECK and note any that are not satisfied or not evaluable.
3. Identify all applicable FAILURE MODES and apply the MOST SEVERE score cap.
4. Select the highest score from the SCORING RUBRIC that does not exceed the active score cap.

TASK:
Assign a final score from 1 to 5 based strictly on the procedure above.

REQUIRED OUTPUT FORMAT (YOU MUST FOLLOW THIS EXACTLY):
Score: [number from 1 to 5]
Explanation: [brief explanation citing concrete visual evidence or lack thereof]

IMPORTANT:
- Base all judgments on observable visual evidence only.
- Do NOT infer intent, plausibility, or realism.
- Functional behavior must be demonstrated by visible effects.
- Score caps MUST be applied before selecting a final score.
"""

    # Make API call to GPT-4o with vision
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": evaluation_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Extract the grade/response
        grade = response.choices[0].message.content.strip()
        
        # Calculate cost from usage information
        usage = response.usage
        cost = _calculate_gpt4o_cost(usage)
        
        return grade, cost
        
    except Exception as e:
        raise Exception(f"Error calling OpenAI API: {e}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Use command-line argument if provided
        image_path = sys.argv[1]
        try:
            result, cost = image_evaluator(image_path)
            print(f"\nEvaluation Result:\n{result}")
            print(f"\nCost: ${cost:.6f}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # If no argument provided, look for images in generated_images/images_to_test
        images_to_test_dir = Path("generated_images/images_to_test")
        
        if not images_to_test_dir.exists():
            print(f"Error: Directory not found: {images_to_test_dir.absolute()}", file=sys.stderr)
            print("Usage: python image_evaluator.py <image_path>")
            print("   OR: Place images in generated_images/images_to_test/")
            sys.exit(1)
        
        # Find all image files in the directory
        # Use case-insensitive matching to avoid duplicates on Windows
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_to_test_dir.glob(ext))
            image_files.extend(images_to_test_dir.glob(ext.upper()))
        
        # Remove duplicates (Windows is case-insensitive, so *.png and *.PNG match the same file)
        image_files = list(set(image_files))
        
        if not image_files:
            print(f"No image files found in: {images_to_test_dir.absolute()}", file=sys.stderr)
            print("Usage: python image_evaluator.py <image_path>")
            print("   OR: Place images in generated_images/images_to_test/")
            sys.exit(1)
        
        # Process all images found
        print(f"Found {len(image_files)} image(s) in {images_to_test_dir.absolute()}\n")
        for i, image_path in enumerate(sorted(image_files), 1):
            print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
            print("-" * 70)
            try:
                result, cost = image_evaluator(image_path)
                print(f"\nEvaluation Result:\n{result}")
                print(f"Cost: ${cost:.6f}\n")
            except Exception as e:
                print(f"Error evaluating {image_path.name}: {e}\n", file=sys.stderr)
            print()

