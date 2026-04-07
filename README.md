# tool-genesis

## Quick Start (Benchmark)

- Prerequisites:
  - `conda`
  - Python 3.10+
  - `OPENAI_API_KEY` (for generation)
  - Optional: `OPENAI_BASE_URL` (e.g. `https://api.openai.com/v1` or your gateway URL)

- Environment setup:
  - Create and activate a conda environment:
    - `conda create -n toolgenesis python=3.10 -y`
    - `conda activate toolgenesis`
  - Install dependencies (pick one):
    - Using pip: `pip install -r requirements.txt`
    - Using uv: `pip install uv && uv pip install -r requirements.txt`

- Configure environment variables:
  - Copy the template and create your local config:
    - `cp .env.template .env`
  - Edit `.env` and fill in your API keys (for a basic run you only need the OpenAI section):
    - At minimum, set `OPENAI_API_KEY`
    - Also set `OPENAI_BASE_URL` (e.g. `https://api.openai.com/v1` or your OpenAI-compatible gateway URL)
    - Other provider keys in `.env.template` are optional and only needed if you plan to use them
  - `.env` is local and should not be committed (it is ignored by `.gitignore`).

- Configure models for benchmark generation:
  - Edit `scripts/run_benchmark/generate_mcp.sh`
  - In the `PLATFORM_MODELS` array:
    - Use model IDs (`model_id`) that are valid for your LLM provider
    - `model_id` is the model name/ID shown on the provider console for the API key you are using
    - Different API keys or providers may expose different model lists and IDs, so make sure you copy the one that matches your own key
    - Example (as provided by default):
      - `OPENAI:openai/gpt-4.1-mini,openai/gpt-4.1,openai/gpt-o3,openai/gpt-5.1,openai/gpt-5.2`
      - `OPENAI:anthropic/claude-sonnet-4`
      - `OPENAI:google/gemini-3-flash-preview`
    - You can comment out or replace entries with the models you actually have access to.

- Default model used in utility LLM calls:
  - See `src/utils/llm.py` function `call_llm`:
    - Default arguments:
      - `model="gpt-4.1-mini"`
      - `platform="openai"`
  - Make sure this `model` value matches a model ID that:
    - Exists on the platform specified by `platform`
    - Is consistent with the model name/ID configured for the API key on that platform
    - Is consistent with your `.env` configuration (API key and base URL for that platform)
  - You can override `model` and `platform` when calling `call_llm` if you need to use a different provider.

- Generate environments (tool_genesis_v3):
  ```bash
  bash scripts/run_benchmark/generate_mcp.sh
  ```

- Run evaluation:
  ```bash
  bash scripts/run_benchmark/run_evaluation.sh
  ```

- Summarize results:
  ```bash
  python3 scripts/run_benchmark/summarize_results.py \
    --path temp/eval_results_v3 \
  ```
