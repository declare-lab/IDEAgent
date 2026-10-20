<p align="center">
  <img src="assets/title_card.png" alt="IDEAgent: Quality-Diversity search for scientific ideation" width="640">
</p>

# 🧠 IDEAgent

**Agentic Quality-Diversity Search for Research Idea Generation**

An agentic quality-diversity (QD) search system for generating research ideas,
plus baseline generation methods used for comparison.

## 📝 Changelog

- **2026-10-20**: Added `verifiable/`: IDEAgent on AlphaResearch and ALE-Bench Lite, the
  four baselines under the same program condition, and the Yield evaluation under the
  three-judge diversity jury.

## 📖 Abstract

> Large Language Models (LLMs) have significantly advanced the automation of scientific
> discovery. However, existing systems share a core limitation: they optimize ideas
> independently for either *Quality* or *Diversity*. This often results in closely
> related ideas or large sets of trivial, unsound, or unclear concepts. We instead argue
> that research ideation should be framed as a conjunction of *Quality* and *Diversity*
> objectives, i.e., a *Quality-Diversity* **(QD)** search. Accordingly, we introduce
> IDEAgent, a *multi-agent* framework that tracks idea evolution through *lineages*. We
> jointly drive *Quality* using multi-objective feedback for repair and refinement,
> while *Diversity* is achieved through *lightweight* sequential memory and explicit
> comparison against completed ideas, historical variants, and rejected proposals. To
> evaluate this joint objective, we develop Yield, a metric that computes the largest
> mutually diverse set of ideas satisfying predefined quality thresholds. Across 32
> topics spanning 8 CS domains, IDEAgent outperforms the best baseline by 3.89× on
> Yield, while achieving Yield of at least 2 on 8× as many topics. We further analyze
> quality improvements, showing that repair and refinement improve logical rigor and
> clarity while preserving non-obviousness. Lastly, we extend IDEAgent to two algorithm
> discovery tasks for verifiable quality judgment and find that it still consistently
> outperforms all baselines, and yields the largest set of high-quality and diverse
> programs that beat a trivial/standard baseline.

## 🗂️ What Is Here

```text
.
├── ideagent/          # runtime package: IDEAgent's search/repair/refinement
│                      # loop, quality + diversity evaluators, shared infra
├── baselines/
│   ├── simple/        # stateless-independent, single-shot batch, and
│   │                  # sequential-memory baselines
│   └── nova/          # NOVA-style closed-book baseline
├── verifiable/        # IDEAgent on verifiable domains (AlphaResearch,
│                      # ALE-Bench): ideas with programs that are run and scored
├── scripts/           # runners and evaluators (run_ideagent.py,
│                      # eval_ideas.py, select_lineage_representatives.py)
├── configs/           # canonical YAML configs for each pipeline
└── data/
    ├── bkgd_papers.jsonl  # topic input data
    └── examples/          # sample generations from IDEAgent and every baseline
```

## ⚙️ Setup

```bash
pip install -e .
```

Set the relevant API keys as environment variables or in a `.env` file
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` — see `ideagent/clients.py`).

## 🚀 Run IDEAgent (main system)

```bash
python scripts/run_ideagent.py --config configs/ideagent.yaml
```

## 🧪 Run a Baseline

```bash
# Stateless-independent or single-shot batch
python baselines/simple/run_simple_baseline.py --config baselines/simple/stateless.yaml
python baselines/simple/run_simple_baseline.py --config baselines/simple/single_shot.yaml

# Sequential-memory
python baselines/simple/sequential_memory.py --config baselines/simple/sequential_memory.yaml

# NOVA closed-book
python baselines/nova/run_nova_closed_book.py --config baselines/nova/nova_closed_book.yaml
```

## 📊 Evaluate

Runs both quality and diversity eval in one command:

```bash
python scripts/eval_ideas.py \
    --config configs/comp_quality_eval_opus47.yaml configs/diversity_eval_opus47.yaml \
    --input <final_idea_cores.jsonl> \
    --quality-output <quality_report.jsonl> \
    --diversity-output <diversity_report.jsonl> \
    --resume
```

Pass `--skip-quality` or `--skip-diversity` to run only one half. `--start-idx`/
`--end-idx` slice the topic list for sharding a run. `--max-workers` overrides
both evaluators' concurrency.

## 🏆 Results

Mean Yield across 32 topics spanning 8 CS domains. Yield(τ) is the size of the
largest mutually-diverse (pairwise diversity ≥ 7) subset of an archive that is
simultaneously sound (≥ 7), clear (≥ 6), and non-obvious (≥ τ); Yield(τ=7) is the
primary metric reported in the paper.

| Method                 | Yield (τ=7, primary) | Yield (τ=6) |
| ----------------------- | :-------------------: | :----------: |
| **IDEAgent (ours)**     | **1.09**               | **2.31**     |
| NOVA Closed-Book        | 0.28                   | 1.16         |
| Stateless-Independent   | 0.25                   | 0.81         |
| Sequential-Memory       | 0.19                   | 0.53         |
| Single-Shot Batch       | 0.00                   | 0.00         |

IDEAgent outperforms the best baseline (NOVA) by **3.89×** on Yield(τ=7).

## 💡 Examples

`data/examples/` holds sample generations from IDEAgent and every baseline, one
JSONL file per method, over the same fixed set of topics for direct comparison.

## 🔬 Verifiable Search

`verifiable/` runs IDEAgent where an idea's quality can be measured: each idea comes
with the complete program that implements it, which is run and scored on
[AlphaResearch](https://github.com/answers111/alpha-research) (9 mathematical
construction problems) and [ALE-Bench](https://github.com/SakanaAI/ALE-Bench) Lite
(7 AtCoder Heuristic Contest problems). Diversity is judged by a three-judge jury.
See [`verifiable/README.md`](verifiable/README.md) for the setup, runs and evaluation.

## 📄 Citation

```bibtex
@misc{gumma2026ideagentagenticqualitydiversitysearch,
      title={IDEAgent: Agentic Quality-Diversity Search for Research Idea Generation}, 
      author={Varun Gumma and Navonil Majumder and Soumitra Sinhahajari and Soujanya Poria},
      year={2026},
      eprint={2607.22375},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.22375}, 
}
```
