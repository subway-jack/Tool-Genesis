# Tool-Genesis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

**Tool-Genesis** is a benchmark for evaluating how well large language models can *create* tools, not just use them. Given a natural-language server specification, an LLM must generate a fully functional MCP (Model Context Protocol) server -- including tool schemas, business logic, state management, and error handling -- then pass a suite of held-out unit tests.

## Key Features

| Metric | Value |
|--------|-------|
| MCP server specifications | 86 |
| Application domains | 24 |
| Ground-truth tools | 508 |
| Benchmark tasks | 2,150 |
| Held-out unit tests | 9,441 (21% negative/boundary) |
| Models evaluated | 19 |

A **4-level diagnostic rubric** (L1--L4) measures progressive difficulty: from single stateless tools (L1), through multi-tool servers (L2), stateful interactions (L3), to complex servers requiring external APIs and sandboxed execution (L4).

## Quick Start

```bash
git clone https://github.com/subway-jack/Tool-Genesis.git
cd Tool-Genesis
pip install -r requirements.txt
cp .env.template .env  # fill in API keys
```

At minimum, set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) in `.env`.

## Running Experiments

### 1. Generate MCP servers

```bash
# Generate MCP servers (Direct strategy)
python scripts/run_benchmark/generate_mcp_from_task.py \
  --data-path data/tool_genesis_v3.json \
  --out-root temp/results \
  --model gpt-4.1 --strategy direct --platform openai
```

Or run the full model sweep:

```bash
bash scripts/run_benchmark/generate_mcp.sh
```

### 2. Evaluate generated servers

```bash
python scripts/run_benchmark/run_evaluation.py \
  --pred-path temp/results/direct_openai_gpt-4-1 \
  --out-root temp/eval_results \
  --workers 1
```

Or evaluate all results at once:

```bash
bash scripts/run_benchmark/run_evaluation.sh
```

### 3. Summarize results

```bash
python scripts/run_benchmark/summarize_results.py \
  --path temp/eval_results_v3
```

## Benchmark Structure (L1--L4)

| Level | Description | Scope |
|-------|-------------|-------|
| **L1** | Single stateless tool | Schema correctness, basic I/O |
| **L2** | Multi-tool stateless server | Tool orchestration, shared utilities |
| **L3** | Stateful server | In-memory state, cross-call consistency |
| **L4** | Complex / API-dependent server | External API mocking, sandboxed execution |

Each level is evaluated independently so that per-level pass rates reveal *where* a model's tool-creation ability breaks down.

## Repository Layout

```
data/               # Benchmark dataset (tool_genesis_v3.json)
scripts/            # Generation, evaluation, and summarization scripts
src/                # Core library (LLM clients, evaluation harness, utilities)
requirements.txt    # Python dependencies
```

## Citation

```bibtex
@article{toolgenesis2025,
  title   = {Tool-Genesis: Evaluating Tool Creation Ability of Large Language Models},
  author  = {Subway Jack and others},
  year    = {2025},
  note    = {Under review}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
