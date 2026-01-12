# Model Comparison and Grading Script

## Overview

`model_comparison_and_grading.py` is a tool for testing multiple image generation models against one or more prompts. It generates images using different models (currently from OpenAI or from FAL.ai), and optionally grades the image generations using GPT-4o vision, and produces detailed reports with cost tracking.

`PROMPTS/prompt_test_set.json` contains the grading criteria.

`model_config.py` contains the model parameters.

The models currently available are: gpt-image-1, DALL·E 3, DALL·E 2, flux-pro/v1.1-ultra, recraft/v3, flux-2, bria/text-to-image/3.2, imagen4/preview/fast, HiDream-I1 full, HiDream-I1 fast.


## Features

- **Multi-Prompt Support**: Test one or multiple prompts across different models
- **Multi-Model Support**: Compare results from multiple image generation models
- **Interactive Selection**: Choose prompts and models interactively or use "all" option
- **Optional Grading**: Automatically evaluate generated images against grading criteria
- **Cost Tracking**: Track both image generation and grading costs separately
- **Comprehensive Reports**: Generate JSON reports with generation times, costs, and grades
- **Organized Output**: Automatically organize images in timestamped directories
- **Quality Selection**: Configure quality settings for OpenAI models per model (not per prompt)

## Prerequisites

- Python 3.7 or higher
- API keys for the models you want to use:
  - OpenAI API key (for OpenAI models and grading)
  - FAL API key (for FAL.ai models)
- Required Python packages (see Installation)

## Installation

1. Install required dependencies:
```bash
py -m pip install -r requirements.txt
```

Required packages:
- `openai>=1.0.0` - For OpenAI image generation and grading
- `fal-client>=0.4.0` - For FAL.ai image generation
- `requests>=2.28.0` - For HTTP requests

2. Set up API keys as environment variables:

**PowerShell (temporary for current session):**
```powershell
$env:OPENAI_API_KEY = "your-openai-api-key-here"
$env:FAL_KEY = "your-fal-api-key-here"
```

**Windows (permanent):**
- Open System Properties → Environment Variables
- Add `OPENAI_API_KEY` and `FAL_KEY` to your user or system variables

## Configuration

### Prompt Test Set

The script reads prompts from `PROMPTS/prompt_test_set.json`. This file should contain either:
- A list of prompt objects, or
- A dictionary with a `"prompts"` key containing a list of prompt objects

Each prompt object should have:
- `prompt_name`: Short identifier for the prompt
- `prompt`: The actual prompt text to send to the model
- `grading_criteria` (optional): Dictionary containing evaluation criteria for grading

Example prompt structure:
```json
{
  "prompt_name": "example_prompt",
  "prompt": "A beautiful sunset over mountains",
  "grading_criteria": {
    "evaluation_instructions": "...",
    "hard_fail_conditions": [...],
    "required_checks": [...],
    "failure_modes": {...},
    "scoring_rubric": {...}
  }
}
```

### Model Configuration

Models are configured in `model_config.py`. The script automatically reads from the `MODELS` list. Each model configuration includes:
- `model_name`: Display name
- `provider`: Either "openai" or "fal"
- `request`: API call configuration
- `defaults`: Default parameters (quality, size, etc.)
- `available_qualities`: List of available quality options (for OpenAI)

## Usage

### Basic Usage

Run the script:
```bash
py model_comparison_and_grading.py
```

### Interactive Workflow

1. **Select Prompts**: 
   - The script displays all available prompts with previews
   - Enter numbers separated by commas (e.g., `1,3,5`) or `all` for all prompts
   - Example: `1,3,5` or `all`

2. **Select Models**:
   - The script displays all available models with provider information
   - Enter numbers separated by commas (e.g., `2,4`) or `all` for all models
   - Example: `2,4` or `all`

3. **Configure Quality (OpenAI models only)**:
   - For each OpenAI model that supports quality settings, you'll be prompted to select a quality level
   - This setting applies to all prompts for that model
   - If only one quality option exists, it's selected automatically

4. **Enable Grading**:
   - Choose whether to grade the generated images (y/n)
   - If enabled, images are evaluated using GPT-4o vision against the grading criteria in the prompt configuration
   - Prompts without grading criteria will be skipped during evaluation

5. **Generation Process**:
   - The script generates images for each prompt-model combination
   - Progress is displayed with task numbers and status
   - Each generation shows: model name, provider, generation time, and cost

### Non-Interactive Mode

If no interactive input is detected (e.g., when piped or automated):
- First prompt is selected by default
- First model is selected by default
- Grading is disabled by default

## Output Structure

### Directory Organization

The script creates a timestamped directory in `generated_images/`:

**Single Prompt:**
```
generated_images/
  └── PromptName_2026-01-12T12-46-03-329Z/
      ├── image_1_model1.png
      ├── image_1_model2.png
      └── image_generation_report.json
```

**Multiple Prompts:**
```
generated_images/
  └── multi_prompt_test_2026-01-12T12-46-03-329Z/
      ├── PromptName1/
      │   ├── image_1_model1.png
      │   └── image_1_model2.png
      ├── PromptName2/
      │   ├── image_1_model1.png
      │   └── image_1_model2.png
      └── image_generation_report.json
```

### Report Format

The `image_generation_report.json` file contains an array of generation records:

```json
[
  {
    "prompt_name": "example_prompt",
    "model_name": "gpt-image-1",
    "generation_time": 2.45,
    "cost": 0.0400,
    "grade": "Detailed evaluation text...",
    "grading_cost": 0.000150
  },
  {
    "prompt_name": "example_prompt",
    "model_name": "fal-model-1",
    "generation_time": 3.12,
    "cost": 0.0200
  },
  {
    "prompt_name": "another_prompt",
    "model_name": "gpt-image-1",
    "generation_time": null,
    "cost": null,
    "error": "API rate limit exceeded"
  }
]
```

Report fields:
- `prompt_name`: Name of the prompt used
- `model_name`: Name of the model used
- `generation_time`: Time taken in seconds (rounded to 2 decimals)
- `cost`: Generation cost in USD (rounded to 4 decimals, null if unavailable)
- `grade`: Evaluation text (only if grading enabled and criteria available)
- `grading_cost`: Cost of grading in USD (only if grading enabled)
- `error`: Error message if generation failed

### Cost Summary

At the end of execution, the script displays:
- Total image generation cost
- Total grading cost (if grading enabled)
- Combined total cost

## Examples

### Example 1: Compare Two Models on One Prompt

```bash
py model_comparison_and_grading.py
# Select prompt: 1
# Select models: 1,2
# Quality selection: (if OpenAI models)
# Grading: y
```

### Example 2: Test All Models on All Prompts

```bash
py model_comparison_and_grading.py
# Select prompt: all
# Select models: all
# Quality selection: (for each OpenAI model)
# Grading: n
```

### Example 3: Single Prompt, Single Model, No Grading

```bash
py model_comparison_and_grading.py
# Select prompt: 1
# Select models: 1
# Quality selection: (if OpenAI)
# Grading: n
```

## Error Handling

- **Missing API Keys**: The script will prompt for API keys if not found in environment variables
- **Generation Failures**: Errors are caught and logged in the report; execution continues with other combinations
- **Grading Failures**: If grading fails for an image, an error message is stored in the report
- **Missing Grading Criteria**: Prompts without grading criteria are skipped during evaluation (with a warning)
- **Invalid Selections**: Out-of-range selections are ignored with warnings

## Troubleshooting

### "Prompt test set file not found"
- Ensure `PROMPTS/prompt_test_set.json` exists in the project root
- Check that the file path is correct

### "No models available in configuration"
- Check that `model_config.py` contains a `MODELS` list with at least one model
- Verify the file is in the same directory as the script

### "Missing dependency `openai`" or "Missing dependency `fal-client`"
- Install missing packages: `py -m pip install -r requirements.txt`

### "Missing environment variable `OPENAI_API_KEY`"
- Set the API key as an environment variable (see Installation section)
- Or the script will prompt you to enter it interactively

### Images not generating
- Check API key validity
- Verify model configuration in `model_config.py`
- Check network connectivity
- Review error messages in the console output and report file

### Grading not working
- Ensure grading was enabled (answered 'y' when prompted)
- Verify prompts have `grading_criteria` in the JSON file
- Check that `OPENAI_API_KEY` is set and valid
- Review error messages in the console output

## Dependencies

The script relies on several utility modules:
- `general_utilities.image_generation_utils`: Image generation functions for OpenAI and FAL
- `general_utilities.image_script_utils`: Directory creation, timestamping, API key management
- `image_evaluator`: Image grading using GPT-4o vision
- `model_config`: Model configuration definitions

## Notes

- Quality settings for OpenAI models are selected once per model and apply to all prompts
- The script supports both single and multiple prompt testing with appropriate directory organization
- Cost tracking may not be available for all models (FAL pricing may require API access)
- Grading uses GPT-4o vision API and incurs additional costs
- All file paths use safe filename conversion to avoid filesystem issues

## Exit Codes

- `0`: Success
- `1`: Error during execution
- `130`: Cancelled by user (Ctrl+C)

