"""Settings of the verifiable search: models and serving, budgets, quality tie-breaks and the bug-fix loop."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from . import tasks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                    # verifiable
ARCHIVE = ROOT.parent                 # the IDEAgent repository
NATIVE_QD_CONFIG = ARCHIVE / "configs" / "ideagent.yaml"
VPY = os.environ.get("VLLM_PYTHON", sys.executable)   # interpreter with vLLM installed (serves the models)

TASKS = tasks.SUITES["alpha_research"]   # the 9 AlphaResearch problems (default task list)
SUITES = tasks.SUITES                    # --tasks also accepts a suite name: alpha_research, ale_bench
CONDITION = "program_qd_v7"

# Models are served with vLLM from their Hugging Face ids; "gpu" is the CUDA_VISIBLE_DEVICES value of each server and
# "memory" its vLLM GPU memory fraction: edit both for your machine. The FP8 checkpoints also run on GPUs without FP8
# arithmetic (e.g. A100), where vLLM serves them weight-only.
# Evaluator (rubric, soundness panel, judges, critics, steno): a 35B MoE with 3B active parameters.
QWEN36 = {"name": "qwen36", "family": "qwen", "model": "Qwen/Qwen3.6-35B-A3B-FP8",
          "revision": "95a723d08a9490559dae23d0cff1d9466213d989", "port": 8011, "memory": 0.90,
          "max_num_seqs": 64, "max_num_batched_tokens": 16384, "gpu": "0", "max_model_len": 131072}
# Ideator: every draft waits on its thinking, so give it the most GPUs (e.g. "gpu": "1,2,3" with "dp": 3). MTP
# speculative decoding keeps the model's sampling distribution (vLLM verifies every drafted token).
MTP = '{"method":"mtp","num_speculative_tokens":2}'
QWEN38 = {"name": "qwen38", "family": "qwen", "model": "Qwen/Qwen3.8-27B-FP8",
          "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a", "port": 8016, "memory": 0.90,
          "max_num_seqs": 32, "max_num_batched_tokens": 16384, "gpu": "1", "max_model_len": 131072, "speculative": MTP}
IDEATOR_REPLICAS = (QWEN38,)            # topic i -> IDEATOR_REPLICAS[i % len]
# Diversity jury (yield_eval.py): three model families independent of the ideator and evaluator, each served alone
# while no search server runs, at its model card's thinking settings.
GEMMA = {"name": "gemma4", "family": "gemma", "model": "google/gemma-4-31B-it", "port": 8031, "memory": 0.90,
         "max_num_seqs": 64, "max_num_batched_tokens": 16384, "gpu": "0,1,2,3", "tp": 2, "dp": 2, "max_model_len": 131072,
         "reasoning_parser": "gemma4"}
GEMMA_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
M27 = {"name": "m27", "family": "minimax", "model": "MiniMaxAI/MiniMax-M2.7", "port": 8051, "memory": 0.90,
       "max_num_seqs": 64, "max_num_batched_tokens": 16384, "gpu": "0,1,2,3", "tp": 4, "max_model_len": 131072,
       "reasoning_parser": "minimax_m2"}
M27_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 40}
GLM45AIR = {"name": "glm45air", "family": "glm", "model": "zai-org/GLM-4.5-Air-FP8", "port": 8061, "memory": 0.90,
            "max_num_seqs": 8, "max_num_batched_tokens": 8192, "gpu": "0,1,2,3", "tp": 4, "max_model_len": 32768,
            "reasoning_parser": "glm45"}
GLM45AIR_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 40}
EVALUATOR = QWEN36
SERVERS = (EVALUATOR,) + IDEATOR_REPLICAS

# Model-card sampling. The native clients forward only temperature/top_p/top_k, so the overlay adds Qwen3.6's
# presence_penalty (keyed by temperature) and the ideator's reasoning effort.
QWEN36_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
QWEN36_NON_THINKING = {"temperature": 0.7, "top_p": 0.80, "top_k": 20}
QWEN38_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
PRESENCE_PENALTY = {QWEN36["model"]: {1.0: 1.5, 0.7: 1.5}}
REASONING_EFFORT = {QWEN38["model"]: "medium"}     # at the default (xhigh) most drafts exhaust the budget thinking

# Token budgets, thinking included.
ROLE_SETTINGS = {
    "ideator": {"enable_thinking": True, "max_new_tokens": 32768},
    "critic": {"enable_thinking": True, "max_new_tokens": 32768},
    "quality": {"enable_thinking": True, "max_new_tokens": 24576},
    "judge": {"enable_thinking": True, "max_new_tokens": 16384},
    "steno": {"enable_thinking": False, "max_new_tokens": 1024},
}
# Every free-text thinking call returns an answer (overlay.install_client_guards): an answer the model ended early is
# continued, and thinking that used the whole budget is closed with Qwen's documented line and answered.
COMPLETE_ANSWERS = True
# Compact JSON under grammar-constrained decoding.
STRUCTURED_OUTPUTS = '{"backend":"xgrammar","disable_any_whitespace":true}'

# Soundness panel: one judgment, fatal at <= 3, checking only that the program implements the proposed method and that
# both are meant for this task; quality comes from the measured run.
SOUNDNESS_JUDGMENTS = 1

# Bug fixing: a program that crashes with a programming bug gets up to CRASH_FIX_ATTEMPTS single-turn ideator calls
# (only the current code and its traceback) to fix it without changing the algorithm, separate from repair and
# refinement. One call may change at most CRASH_FIX_MAX_CHANGED_LINES non-blank lines, all calls together at most
# CRASH_FIX_MAX_TOTAL_CHANGED_LINES of the original program. A bug that survives every call discards the draft.
CRASH_FIX_ATTEMPTS = 10
CRASH_FIX_MAX_CHANGED_LINES = 20
CRASH_FIX_MAX_TOTAL_CHANGED_LINES = 40
CRASH_FIX_SETTINGS = {"enable_thinking": True, "max_new_tokens": 16384, "reasoning_effort": "low"}

IDEATION_BUDGET_S = 60     # wall-clock seconds per program run during the search (split across instances)
FINAL_BUDGET_S = 300       # AlphaResearch's evaluator timeout; one final run per frozen proposal
# Q = performance + NB_TIEBREAK * non-obviousness - RUNTIME_TIEBREAK * total solve seconds. Performance is an integer,
# so non-obviousness only breaks ties, and one point of it outweighs any runtime difference.
NB_TIEBREAK = 1e-3
RUNTIME_TIEBREAK = 1e-6
# Must stay > 0: the native critics see the run notes only when the feasibility press is on (the overlay replaces the
# press itself).
PRESS_FEASIBILITY_TARGET = 60
FINAL_WORKERS = 24         # final runs in parallel, one CPU thread each


def ideator_server(topic_index: int = 0) -> dict:
    return IDEATOR_REPLICAS[topic_index % len(IDEATOR_REPLICAS)]
