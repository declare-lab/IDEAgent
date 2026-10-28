<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/title_card_dark.png">
    <img src="assets/title_card.png" alt="IDEAgent: Quality-Diversity search for scientific ideation" width="640">
  </picture>
</p>

## 📝 Changelog

- **2026-10-28**: Added request-level checkpoint/resume to `scripts/run_ideagent.py`
  (`--checkpoint-dir`, `--resume`): an interrupted search replays its saved responses and
  re-sends only unfinished requests ([#3](https://github.com/declare-lab/IDEAgent/pull/3)).
- **2026-10-20**: Added `verifiable/`: IDEAgent on AlphaResearch and ALE-Bench, the
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

### Resume interrupted API requests

Start a run with a request journal:

```bash
python scripts/run_ideagent.py --config configs/ideagent.yaml --checkpoint-dir runs/my-search
```

After interruption, run the same command with `--resume`:

```bash
python scripts/run_ideagent.py --config configs/ideagent.yaml --checkpoint-dir runs/my-search --resume
```

Every API response is saved to `requests.sqlite` in the checkpoint directory as it
arrives, and outputs go to `data.output_dir` as usual. On `--resume`, topics that
already finished are skipped, and the unfinished topic is rebuilt from the saved
responses without calling the API or logging anything again. The run then continues
from where it stopped, adding to the same outputs.

If a role's model or generation settings (`temperature`, `max_new_tokens`, thinking,
...) changed since the run started, `--resume` stops with an error naming the change:
undo it to resume, or start afresh without `--resume` in a new `--checkpoint-dir`.

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

### Open-ended ideation

Mean Yield across 32 topics spanning 8 CS domains. Yield(τ) is the size of the
largest mutually-diverse (pairwise diversity ≥ 7) subset of an archive that is
simultaneously sound (≥ 7), clear (≥ 6), and non-obvious (≥ τ); Yield(τ=7) is the
primary metric reported in the paper.

| Method              | Yield (τ=7, primary) | Yield (τ=6) |
| ------------------- | :------------------: | :---------: |
| Sequential-Memory   | 0.19                 | 0.53        |
| One-Shot            | 0.00                 | 0.00        |
| Stateless           | 0.25                 | 0.81        |
| NOVA                | 0.28                 | 1.16        |
| **IDEAgent**        | **1.09**             | **2.31**    |

IDEAgent outperforms the best baseline (NOVA) by **3.89×** on Yield(τ=7).

### Verifiable search

Means over the problems of each benchmark (9 for AlphaResearch, 7 for ALE-Bench).
$Q_i$ is 100 × an idea's progress from the trivial baseline to the reference (the
human-best for AlphaResearch, the contest winner for ALE-Bench); $Q_{\max}$ and
$Q_{\text{avg}}$ are the highest and the mean $Q_i$ among a method's ten final ideas.
$Y_k$ is the size of the largest set of mutually distinct ideas, under the three-judge
jury, that beat the baseline with $Q_i \ge k$. See [`verifiable/README.md`](verifiable/README.md).

<table>
  <thead>
    <tr><th rowspan="2" align="left">Method</th><th colspan="6">AlphaResearch</th><th colspan="6">ALE-Bench</th></tr>
    <tr><th>Q<sub>max</sub></th><th>Q<sub>avg</sub></th><th>Y<sub>0</sub></th><th>Y<sub>50</sub></th><th>Y<sub>75</sub></th><th>Y<sub>100</sub></th><th>Q<sub>max</sub></th><th>Q<sub>avg</sub></th><th>Y<sub>0</sub></th><th>Y<sub>25</sub></th><th>Y<sub>50</sub></th><th>Y<sub>75</sub></th></tr>
  </thead>
  <tbody>
    <tr><td>Sequential-Memory</td><td align="center">92.6</td><td align="center">46.1</td><td align="center">4.00</td><td align="center">3.11</td><td align="center">1.67</td><td align="center">0.56</td><td align="center">38.3</td><td align="center">18.6</td><td align="center">2.29</td><td align="center">1.71</td><td align="center">1.14</td><td align="center">0.29</td></tr>
    <tr><td>One-Shot</td><td align="center">53.2</td><td align="center">16.9</td><td align="center">2.78</td><td align="center">1.00</td><td align="center">0.44</td><td align="center">0.33</td><td align="center">14.8</td><td align="center">6.2</td><td align="center">1.14</td><td align="center">0.43</td><td align="center">0.43</td><td align="center">0.00</td></tr>
    <tr><td>Stateless</td><td align="center">93.5</td><td align="center">47.8</td><td align="center">4.11</td><td align="center">3.11</td><td align="center">1.78</td><td align="center">0.56</td><td align="center">45.1</td><td align="center"><b>22.7</b></td><td align="center">2.29</td><td align="center">1.71</td><td align="center">0.57</td><td align="center">0.00</td></tr>
    <tr><td>NOVA</td><td align="center">75.4</td><td align="center">34.4</td><td align="center">4.56</td><td align="center">2.67</td><td align="center">0.78</td><td align="center">0.22</td><td align="center">37.3</td><td align="center">18.0</td><td align="center">2.29</td><td align="center">2.00</td><td align="center">1.00</td><td align="center">0.00</td></tr>
    <tr><td><b>IDEAgent</b></td><td align="center"><b>106.2</b></td><td align="center"><b>54.5</b></td><td align="center"><b>5.67</b></td><td align="center"><b>4.44</b></td><td align="center"><b>3.11</b></td><td align="center"><b>0.67</b></td><td align="center"><b>56.4</b></td><td align="center">22.6</td><td align="center"><b>3.43</b></td><td align="center"><b>2.57</b></td><td align="center"><b>1.57</b></td><td align="center"><b>0.57</b></td></tr>
  </tbody>
</table>

## 💡 Examples

`data/examples/` holds sample generations from IDEAgent and every baseline, one
JSONL file per method, over the same fixed set of topics for direct comparison.

## 🔬 Verifiable Search

`verifiable/` runs IDEAgent where an idea's quality can be measured: each idea comes
with the complete program that implements it, which is run and scored on
[AlphaResearch](https://github.com/answers111/alpha-research) (9 mathematical
construction problems) and [ALE-Bench](https://github.com/SakanaAI/ALE-Bench)
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
