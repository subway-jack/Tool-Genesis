# Tool-Genesis Dataset Card (v3)

## Dataset Description

**Tool-Genesis v3** provides 86 MCP (Model Context Protocol) server specifications for evaluating requirement-driven tool creation. The journal version frames this setting as **latent contract recovery**: recovering interface, implementation, and task-utility contracts from incomplete requirements. Each specification provides a natural-language requirement, evaluator reference tool schemas, synthesized task prompts, and held-out unit-test tuples. A model must implement an MCP server from the requirement without seeing the reference schema, source code, hidden tests, or downstream trajectories during generation.

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Server specifications | 86 |
| Application domains | 24 |
| Ground-truth tools | 508 |
| Synthesized task prompts | 1,720 |
| Unit tests | 9,441 |
| Negative / boundary tests | ~21% |

The public compact JSON includes synthesized task prompts but does **not** include full solver trajectories or exact paper-version L4 replay records. Exact manuscript L4 replay requires retained paper-version task/trajectory assets plus a configurable LLM solver/judge endpoint.

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
| `task_example` | list[string] | Synthesized task prompts for downstream utility evaluation |
| `tool_definitions` | list[object] | Ground-truth MCP tool schemas (name, description, parameters, return type) |
| `unit_test` | dict[string, list] | Public unit-test tuples keyed by tool name; each test contains `function_name`, `arguments`, and expected `function_output_content` |
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

The released unit-test tuples were internally QA-checked for consistency, ordinary cases, and edge/failure cases. The public JSON stores final input-output tuples rather than internal QA logs or per-test provenance/category labels.

## Intended Use

The dataset is designed for:

- Benchmarking LLM requirement-driven MCP tool creation across the L1--L4 diagnostic rubric.
- Comparing generation strategies such as Direct generation and agentic coding plus L1 repair.
- Studying interface recovery, functional correctness, downstream task utility, and failure localization.

## Limitations

- **English only.** All specifications and tests are in English.
- **Exact L4 replay boundary.** Public artifacts support L1--L3 reproduction from requirements, schemas, source, and unit-test tuples; exact paper-version L4 replay requires retained task/trajectory assets and an LLM solver/judge endpoint.
- **No production hardening claim.** Generated servers are evaluation artifacts and should not be treated as production-ready implementations.
- **Limited state and credentials.** Credentialed, network-dependent, persistent production deployments are outside the current benchmark scope.
- **Snapshot in time.** The registry crawl reflects August--September 2025; newer MCP servers are not included.

## License

Benchmark packaging, evaluation code, and derived metadata are released under the [MIT License](../LICENSE). Third-party MCP server artifacts retain their original upstream licenses where applicable.

## Citation

```bibtex
@article{toolgenesis2025,
  title   = {Tool-Genesis: Benchmarking Latent Contract Recovery in MCP Tool Creation},
  author  = {Subway Jack and others},
  year    = {2026},
  note    = {Manuscript under review}
}
```
