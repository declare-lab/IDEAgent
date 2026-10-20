# 🔬 Verifiable Search

IDEAgent on problems whose ideas can be checked by running them. Each idea is a short research proposal together with
the complete program that implements it; the program is run and scored, and its measured score is the idea's quality.
Diversity is still judged on the method, as in the open-ended search.

| Benchmark | Problems | Program | Instances per problem | Quality 100 = |
|---|---|---|---|---|
| [AlphaResearch](https://github.com/answers111/alpha-research) (git submodule, commit `e6d6454`) | 9 mathematical construction problems | Python, `solve(problem)` | 1–2 | the human-best |
| [ALE-Bench Lite](https://github.com/SakanaAI/ALE-Bench) (`setup_alebench.sh`) | 7 AtCoder Heuristic Contest problems: ahc008, 011, 015, 024, 026, 039, 046 | C++, stdin → stdout | 50 public test cases | the contest winner's mean score |

## 🗂️ What Is Here

```text
verifiable/
├── program_qd/
│   ├── pipeline.py     # IDEAgent run: prepare -> ideation -> freeze -> final runs -> report
│   ├── overlay.py      # adapts the ideagent package at run time (the package itself is unmodified)
│   ├── prompts.py      # program-task prompts for every role
│   ├── tasks.py        # AlphaResearch tasks, scoring and the simple baselines to beat
│   ├── alebench.py     # ALE-Bench tasks, judged natively with each problem's own tester
│   ├── baselines.py    # the four baselines under the same program condition
│   ├── yield_eval.py   # Yield and quality under the three-judge diversity jury
│   └── config.py       # models, GPUs, budgets
├── third_party/        # AlphaResearch submodule; ALE-Bench data and tools after setup
├── proposals.py        # freezes the final proposals of a run
├── worker.py           # sandboxed program runner (Linux resource limits)
└── setup_alebench.sh
```

## 🧭 How It Works

`program_qd/overlay.py` installs the following on the `ideagent` package and then starts `scripts/run_ideagent.py` with
`configs/ideagent.yaml`, exactly as an open-ended run would:

- **Task background**: the problem, the program contract and a simple baseline to beat (its pseudocode and measured
  score). Every evaluator and critic sees it; the reference value is never shown.
- **Program-task prompts** (`prompts.py`) for the ideator, critics, rubric, soundness panel, diversity judge, memory
  consolidation and steno.
- **Execution**: every draft's program runs in a sandbox (one CPU thread, 4 GiB): 60 s per program during the search
  on AlphaResearch, the contest time limit per test case on ALE-Bench. Its performance replaces the LLM feasibility
  score: 0 for a failed program, 20 for a valid one with no progress over the baseline, else 20 + 80 × the mean
  progress toward the reference (uncapped). Q = performance, with ties broken by non-obviousness, then runtime.
- **Acceptance**: the program is valid and beats the baseline on every instance, the soundness panel (one judgment:
  the program implements the proposed method, and both are meant for this task) is not fatal, and the diversity gate
  passes. A non-fatal miss is repairable, and accepted ideas are refined further for score.
- **Bug fixing**: a program that crashes with a programming bug gets up to 10 single-turn ideator calls (only the code
  and its traceback; at most 20 changed lines per call and 40 in total), separate from repair and refinement.
- **Search**: elite replacement across lineages; a refinement starts from its parent's construction; each of the 10
  frozen slots holds the best version of its seed's drafts.
- **Transport guards** for the Qwen thinking models: an answer ended early is continued, an exhausted thinking budget
  is closed and answered, and malformed JSON is resampled.

**Baselines** (`baselines.py`): stateless, single-shot, sequential memory and closed-book NOVA (2 rounds × 5 ideas), run
by their own runners in `baselines/` under the same program condition: the same task backgrounds, models, sampling and
ideator prompts, 10 frozen proposals per problem, the same bug-fix loop, and one final run per proposal. As in their
native form, they get no evaluator, critic, repair or execution feedback while generating.

**Final runs**: each frozen proposal's program runs once, at 300 s per program on AlphaResearch (its evaluator's
timeout) and at the contest time limit per test case on ALE-Bench.

## 📏 Evaluation

`yield_eval.py` shows every pair of a method's proposals that beat the baseline to three judges from model families
independent of the ideator and evaluator (Gemma 4 31B, MiniMax-M2.7, GLM-4.5-Air), in both orders, with the programs
removed and the method hidden: do they share the same central move? A judge calls a pair distinct only when both
orders say so, and a pair is distinct when 2 of the 3 judges do.

For a problem with instances $J$, let $s^0_j$ be the simple baseline's score on instance $j$, $s^\star_j$ the reference,
and $s_{ij}$ the score of proposal $i$. Its quality is its mean progress from the baseline to the reference (the same
expression holds for minimization; a failed program scores 0):

$$Q_i = \frac{100}{|J|} \sum_{j \in J} \max\left(0, \frac{s_{ij} - s^0_j}{s^\star_j - s^0_j}\right)$$

- $Y_k$: the size of the largest set of mutually distinct proposals that beat the baseline with $Q_i \ge k$.
- $Q_{\max}$ and $Q_{\text{avg}}$: the highest and the mean $Q_i$ over all of a method's proposals.

All three are averaged over the problems of a benchmark.

## ⚙️ Setup

Linux, with a GPU server for the models.

```bash
git submodule update --init                  # AlphaResearch
pip install -e . -r verifiable/requirements.txt
bash verifiable/setup_alebench.sh            # ALE-Bench only; needs g++ and a Rust toolchain
```

Models are served with vLLM from Hugging Face: the ideator is Qwen3.8-27B-FP8 and the evaluator (every other role) is
Qwen3.6-35B-A3B-FP8. Set each server's GPUs in `program_qd/config.py` (`"gpu"`, the `CUDA_VISIBLE_DEVICES` value;
by default the evaluator uses GPU 0 and the ideator GPU 1). If vLLM is installed in another environment, point
`VLLM_PYTHON` at its interpreter.

## 🚀 Run

From the repository root. A run covers one benchmark (`--tasks alpha_research`, `--tasks ale_bench`, or a comma-separated
list of its problems) and writes `verifiable/program_qd/runs/NAME/`.

```bash
# IDEAgent, then the baselines on the same problems
python -m verifiable.program_qd.pipeline --name ar_qd --tasks alpha_research
python -m verifiable.program_qd.baselines --name ar_baselines --qd-run ar_qd

# Yield under the diversity jury (each judge's server is started in turn), then the scores
python -m verifiable.program_qd.yield_eval judge --baselines-run ar_baselines
python -m verifiable.program_qd.yield_eval score --baselines-run ar_baselines --ks 0,50,75,100

# ALE-Bench: the same commands with --tasks ale_bench (the paper reports k = 0, 25, 50, 75)
python -m verifiable.program_qd.pipeline --name ale_qd --tasks ale_bench --hours 72
```

Both runners resume with `--stage` (`pipeline`: ideation, freeze, final, report; `baselines`: generation, freeze,
final, report). Each run's `report.md` summarizes it; `programs/` holds every draft's program and run, `final/` the
final runs, and `logs/<task>/overlay_events.jsonl` one `program_run` event per draft.
