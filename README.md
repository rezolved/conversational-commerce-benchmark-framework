# Conversational Commerce Benchmark

Minimal public artefacts for:

> **Benchmarking Models for Conversational E-Commerce: A Reproducible Evaluation Framework**
>
> Yuri Vorontsov, Diogo S. Carvalho, Anastasia Vorontsov, Anna Platonova,
> Vladimir Gorovoy, Ilya Briskin, Felix Tseitlin, Senka Krivic, Salman Ahmad.
> *GenAIECommerce 2026* (Third Workshop on Agentic and Generative AI for
> E-Commerce), RecSys 2026, Minneapolis. CEUR-WS, CC BY 4.0.

This dump is **jeans-only**: the Amazon Reviews 2023 jeans catalog, 20 frozen
scenarios, and a minimal runner. It does **not** include conversation traces,
judge outputs, model weights, or other product categories.

## Quick start (no API keys)

Hashing embeddings build a local index so you can verify install without a
provider account. Retrieval quality will be weaker than the paper (the paper
used Qwen3-Embedding via an OpenAI-compatible API).

```bash
git clone https://github.com/rezolved/conversational-commerce-benchmark-framework.git
cd conversational-commerce-benchmark-framework
python3 -m venv .venv
source .venv/bin/activate
# CPU torch first — a plain `pip install -r requirements.txt` may pull
# multi-GB NVIDIA wheels that txtai does not need for the hashing smoke path.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export EMBEDDING_PROVIDER=hashing
PYTHONPATH=src python scripts/build_index.py
PYTHONPATH=src python scripts/run_benchmark.py --help
```

`build_index.py` writes under `artifacts/` (gitignored). The 20 frozen
scenarios load from `data/amazon_jeans_sample_20.json`.

## Run a model (needs an LLM)

Copy `.env.example` to `.env` and set any OpenAI-compatible `AGENT_LLM_*`
endpoint. Then:

```bash
cp .env.example .env
# edit .env — never commit it

PYTHONPATH=src python scripts/run_benchmark.py \
    --dialogs data/amazon_jeans_sample_20.json \
    --model Qwen3-32B \
    --runs 1 \
    --limit 1 \
    --skip-judge
```

`--skip-judge` skips the LLM-as-judge step. A full paper-style run also needs
`JUDGE_*` credentials and should omit `--limit`.

### Execution fidelity

Saved transcripts and judge input include assistant tool calls and tool results
by default. This evidence is required to evaluate grounding and add-to-cart
correctness. `--omit-tool-messages` is available for debugging only; scores
from that mode are not comparable because the judge cannot verify claims
against tool output.

The evaluated assistant model also performs the LLM-based re-ranking step
after vector retrieval. The index, tool schema, and prompts are fixed, but the
re-ranker is therefore model-dependent rather than an identical external
component.

The paper's reported scores were produced by Rezolve AI's internal QuePasa
evaluation endpoint. `src/benchmark/judge.py` is a public, OpenAI-compatible
reference implementation of the same five-rubric interface. It supports new
evaluations but cannot exactly reproduce proprietary judge outputs that are
not part of this jeans-only release.

## Five evaluation rubrics

Each conversation is scored 0/1 on:

| Rubric | What it measures |
|--------|------------------|
| **Intent Understanding** | Did the assistant correctly understand the shopping request? |
| **Clarification** | Were clarifying questions appropriate and helpful? |
| **Recommendation** | Were product suggestions relevant to stated preferences? |
| **Add-to-Cart** | Was the cart operation correct and only done when asked? |
| **Grounding** | Did the assistant stay grounded in catalog data (no hallucination)? |

Total score per conversation: 0–5.

## Repository layout

```
data/
  amazon_jeans_shop.json          # Product catalog (Amazon Reviews 2023)
  amazon_jeans_sample_20.json     # 20 frozen jeans scenarios
src/
  benchmark/                      # Conversation loop, customer sim, judge
  retrieval/                      # txtai index, search tool, assistant agent
scripts/
  build_index.py                  # Build txtai index from the jeans catalog
  run_benchmark.py                # End-to-end runner
  generate_scenarios.py           # Generic template only (not the frozen 20)
paper/figures/
  leaderboard.csv                 # Published jeans leaderboard
  rubric_scores.csv               # Per-rubric pass rates
  make_figures.py                 # Figure regeneration helper
tests/
  test_conversation.py            # Tool-transcript fidelity checks
  test_deterministic_hashing.py   # Cross-process hashing checks
```

## Data

The catalog is a derived subset of the public Amazon Reviews 2023 dataset
(Hou et al., 2024). See [DATA.md](DATA.md).

## Offline verification

After installing dependencies, run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Citation

```bibtex
@inproceedings{vorontsov2026benchmarking,
  title     = {Benchmarking Models for Conversational E-Commerce:
               A Reproducible Evaluation Framework},
  author    = {Vorontsov, Yuri and Carvalho, Diogo S. and Vorontsov, Anastasia
               and Platonova, Anna and Gorovoy, Vladimir and Briskin, Ilya
               and Tseitlin, Felix and Krivic, Senka and Ahmad, Salman},
  booktitle = {Proceedings of the Third Workshop on Agentic and Generative AI
               for E-Commerce (GenAIECommerce 2026)},
  year      = {2026}
}
```

## License

Code is licensed under the [Apache License 2.0](LICENSE). Catalog
redistribution terms are described in [DATA.md](DATA.md).
