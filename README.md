# Tool-Genesis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

**Tool-Genesis** is a benchmark for evaluating how well large language models can *create* tools, not just use them. Given a natural-language server specification, an LLM must generate a functional MCP (Model Context Protocol) server -- including tool schemas and executable implementation logic -- then pass a suite of held-out unit tests. The journal version frames this setting as **latent contract recovery**: recovering interface, implementation, and task-utility contracts from incomplete requirements.

## Key Features

| Metric | Value |
|--------|-------|
| MCP server specifications | 86 |
| Application domains | 24 |
| Ground-truth tools | 508 |
| Synthesized task prompts | 1,720 |
| Held-out unit tests | 9,441 (21% negative/boundary) |
| Public compact JSON | `data/tool_genesis_v3.json` |

A **4-level diagnostic rubric** (L1--L4) measures progressive diagnostic stages: L1 surface compliance, L2 interface fidelity, L3 functional correctness, and L4 downstream task utility. The public compact JSON includes requirements, tool schemas, synthesized task prompts, and unit-test tuples. Exact paper-version L4 replay requires retained task/trajectory assets and a configurable LLM solver/judge endpoint.

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
| **L1** | Surface compliance | Server launch and tool-registry exposure |
| **L2** | Interface fidelity | Schema-F1 against evaluator reference schemas |
| **L3** | Functional correctness | Held-out unit tests with ordinary and edge/failure cases |
| **L4** | Downstream task utility | Proxy-agent task solving with generated tools |

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
  title   = {Tool-Genesis: Benchmarking Latent Contract Recovery in MCP Tool Creation},
  author  = {Subway Jack and others},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
