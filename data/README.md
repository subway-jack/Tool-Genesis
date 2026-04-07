# Tool-Genesis Dataset Card (v3)

## Dataset Description

**Tool-Genesis v3** provides 86 MCP (Model Context Protocol) server specifications for evaluating the tool-creation ability of large language models. Each specification describes a realistic server -- complete with natural-language requirements, ground-truth tool schemas, example tasks, and held-out unit tests -- that an LLM must implement from scratch.

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Server specifications | 86 |
| Application domains | 24 |
| Ground-truth tools | 508 |
| Benchmark tasks | 2,150 |
| Unit tests | 9,441 |
| Negative / boundary tests | ~21% |

## Fields

Each entry in `tool_genesis_v3.json` is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `server_id` | string or null | Unique identifier (may be null for anonymous sources) |
| `server_name` | string | Human-readable server name |
| `server_slug` | string | URL-safe slug used as a directory/file key |
| `primary_label` | string | Primary domain category (e.g., "Web Search & Research") |
| `secondary_labels` | list[string] | Additional domain tags from the taxonomy |
| `agent_input_prompt` | string | Natural-language requirement specification given to the LLM |
| `task_example` | list[string] | Illustrative tasks the server should support |
| `tool_definitions` | list[object] | Ground-truth MCP tool schemas (name, description, parameters, return type) |
| `unit_test` | dict[string, list] | Held-out unit tests keyed by tool name; each test contains `function_name`, `arguments`, `expected_output`, and metadata |
| `server_class` | string | Complexity class: `stateless` or `stateful` |
| `requires_api` | bool | Whether the server requires external API access |
| `sandbox_level` | string | Diagnostic level (`L1`--`L4`) assigned to the server |

## Domain Taxonomy

The 24 primary domains span areas such as Web Search & Research, API Integration, Data Processing, Finance, Healthcare, Education, DevOps, and more. Each server is assigned exactly one `primary_label` and zero or more `secondary_labels`.

## Data Collection

**Sources.** Server specifications were crawled from four registries between August and September 2025:

- GLMA (General-purpose LLM App registry)
- Smithery
- GitHub (public MCP server repositories)
- HuggingFace (MCP-related datasets and model cards)

**Filtering pipeline.** A four-stage pipeline reduced the candidate pool to the final benchmark:

| Stage | Servers remaining |
|-------|-------------------|
| Initial crawl | 572 |
| De-duplication and format validation | 401 |
| Complexity and coverage filtering | 212 |
| Executability and annotation pass | 134 |
| Final benchmark (after quality review) | 86 |

## Quality Assurance

Three independent annotators reviewed every server specification for correctness, completeness, and consistency of unit tests. Inter-annotator agreement was measured at Cohen's kappa = 0.85, indicating strong agreement. Disagreements were resolved by majority vote and a subsequent reconciliation pass.

## Intended Use

The dataset is designed for:

- Benchmarking LLM tool-creation capabilities across the L1--L4 diagnostic rubric.
- Comparing generation strategies (direct, chain-of-thought, iterative refinement).
- Studying how model scale and architecture affect tool-creation performance.

## Limitations

- **English only.** All specifications and tests are in English.
- **No credentialed servers.** Servers requiring real API credentials (OAuth tokens, paid-tier keys) are excluded; `requires_api` servers use mock/stub backends.
- **No persistent state.** Stateful servers (L3/L4) use in-memory state only; no database or filesystem persistence is tested.
- **Snapshot in time.** The registry crawl reflects August--September 2025; newer MCP servers are not included.

## License

This dataset is released under the [MIT License](../LICENSE).

## Citation

```bibtex
@article{toolgenesis2025,
  title   = {Tool-Genesis: Evaluating Tool Creation Ability of Large Language Models},
  author  = {Subway Jack and others},
  year    = {2025},
  note    = {Under review}
}
```
