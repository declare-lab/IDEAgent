"""Yield of a program QD run and its baselines run under the downstream diversity jury (the paper's evaluation).

Per task and method, every pair of frozen proposals whose final program beats the baseline on every instance is shown
to three judges from model families independent of the ideator and evaluator (config.GEMMA, config.M27,
config.GLM45AIR), in both orders: "same central move?" (full method texts, programs removed, method hidden). A judge
calls a pair distinct only when both orders say so; a pair is distinct when 2 of the 3 judges do.
  Y_k: the size of the largest set of mutually distinct proposals that beat the baseline with Q_i >= k, where Q_i is
       100 x the mean progress from the baseline to the best-known value (floored at 0); mean over tasks.
  Q_max, Q_avg: the highest and the mean Q_i over all of a method's frozen proposals (a failed program scores 0).

    python -m verifiable.program_qd.yield_eval judge --baselines-run NAME [--judges gemma,minimax,glm]
    python -m verifiable.program_qd.yield_eval score --baselines-run NAME [--ks 0,50,75,100]
(run from the repository root; judge starts each judge's server in turn, so no other server may hold its GPUs)
Writes runs/NAME/yield/<judge>/calls.jsonl (resumable).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import re
import threading

import numpy as np

from verifiable import proposals
from verifiable.common import load_json, now

from . import baselines, config, overlay, pipeline, tasks

JUDGES = {"gemma": (config.GEMMA, config.GEMMA_THINKING, 32768), "minimax": (config.M27, config.M27_THINKING, 32768),
          "glm": (config.GLM45AIR, config.GLM45AIR_THINKING, 8192)}       # server, sampling, max new tokens
SYSTEM = (
    "You are a neutral expert. You see two research proposals, A and B, for ONE fixed mathematical construction or "
    "optimization problem (background below). Decide whether they share the same central move: proposals share a "
    "central move when their methods are essentially the same idea, meaning the same construction or parametrization of "
    "the solution, searched or optimized in essentially the same way, even if they are worded, tuned or implemented "
    "differently. Methods that rest on different constructions, parametrizations or search principles do not. Judge the "
    "method only, not its quality, length or wording. Return strict JSON: {\"same\": true or false, \"reason\": \"<one "
    "short sentence>\"}.\n\nBACKGROUND:\n")
SCHEMA = {"type": "json_schema", "json_schema": {"name": "pair", "schema": {
    "type": "object", "required": ["same", "reason"],
    "properties": {"same": {"type": "boolean"}, "reason": {"type": "string"}}}}}


def method_text(idea_text: str) -> str:
    """The proposal without its program, in full."""
    return re.sub(r"```.*?(```|$)", "", idea_text or "", flags=re.S).strip()


def quality(task: str, scores, baseline) -> float:
    """Q_i: 100 x the mean progress from the baseline to the best-known value (tasks.performance without its 20-point
    offset and rounding); 0 for a failed program."""
    if not scores or any(s is None for s in scores):
        return 0.0
    progress = []
    for value, base, best in zip(scores, baseline, tasks.TASKS[task]["reference"]):
        span = (best - base) if tasks.TASKS[task]["direction"] == "max" else (base - best)
        gain = (value - base) if tasks.TASKS[task]["direction"] == "max" else (base - value)
        progress.append(max(0.0, gain / span) if span > 0 else float(gain > 0))
    return 100 * sum(progress) / len(progress)


def load(run_dir: Path) -> tuple[dict, dict, list[str]]:
    """{task: {method: [{id, text, q, ok}]}} over every frozen proposal (ok: beats the baseline on every instance and
    has a method text), the task backgrounds, and the methods."""
    manifest = load_json(run_dir / "manifest.json")
    qd_run, base = Path(manifest["qd_run"]), manifest["baseline_scores"]
    tasks.use_language(manifest["tasks"])
    background = {t["topic_id"]: t["background_papers"][0]["full_text"]
                  for t in map(json.loads, (qd_run / "topics.jsonl").read_text().splitlines()) if t}
    data: dict = {}
    for method, folder in [("qd", qd_run)] + [(f, run_dir) for f in manifest["frameworks"]]:
        finals = load_json(folder / "final_results.json")["results"]
        for p in proposals.load_frozen(folder / "frozen" / "proposals.json")["proposals"]:
            t = p["task"]
            if p["framework"] != method or t not in base:
                continue
            f = finals.get(p["proposal_id"]) or {}
            valid = bool(f.get("all_valid"))
            raw = p.get("idea_text") or ""
            text = "" if raw == baselines.EMPTY_IDEA else method_text(raw)
            beats = valid and all(tasks.better(t, s, b) for s, b in zip(f["scores"], base[t]))
            data.setdefault(t, {}).setdefault(method, []).append(
                {"id": f"{method}:{p['proposal_id']}", "text": text,
                 "q": quality(t, f["scores"], base[t]) if valid else 0.0, "ok": beats and bool(text)})
    return data, background, ["qd"] + manifest["frameworks"]


def judge(name: str, run_dir: Path) -> None:
    data, background, _ = load(run_dir)
    path = run_dir / "yield" / name / "calls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["a"], r["b"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    todo = [(t, a, b) for t, by in data.items() for items in by.values()
            for x, y in itertools.combinations([i for i in items if i["ok"]], 2) for a, b in ((x, y), (y, x))
            if (a["id"], b["id"]) not in done]
    print(f"{name}: {len(todo)} calls to make", flush=True)
    if not todo:
        return
    from ideagent.clients import build_client
    spec, sampling, max_new_tokens = JUDGES[name]
    sampling = {**sampling, "enable_thinking": True, "max_new_tokens": max_new_tokens}
    overlay.install_client_guards(lambda kind, **fields: None)        # JSON resampling, as in the search
    server = pipeline.launch(spec, run_dir / "yield")
    lock = threading.Lock()
    try:
        def new_client():                    # one per call: the guards read per-client response metadata
            return build_client(model_id=spec["model"], vllm_port=spec["port"], api_key_env="VLLM_API_KEY",
                                request_timeout=3 * 3600)

        def call(job):
            t, a, b = job
            messages = [{"role": "system", "content": SYSTEM + background[t]},
                        {"role": "user", "content": f"[A]\n{a['text']}\n\n[B]\n{b['text']}"}]
            same, reason = None, ""
            for _ in range(3):                 # an unparsable answer is asked again; after 3 it counts as same
                raw = new_client().generate(messages, **sampling, response_format=SCHEMA)
                try:
                    r = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                    if isinstance(r.get("same"), bool):
                        same, reason = r["same"], str(r.get("reason", ""))[:300]
                        break
                except (ValueError, TypeError):
                    pass
            with lock, path.open("a") as handle:
                handle.write(json.dumps({"at": now(), "judge": spec["model"], "task": t, "a": a["id"], "b": b["id"],
                                         "same": same, "reason": reason}) + "\n")

        with ThreadPoolExecutor(max_workers=spec["max_num_seqs"] * spec.get("dp", 1)) as pool:
            list(pool.map(call, todo))
    finally:
        pipeline.stop(server)


def score(run_dir: Path, ks: list[int]) -> None:
    data, _, methods = load(run_dir)
    verdict = {n: {(r["a"], r["b"]): r["same"]
                   for r in map(json.loads, (run_dir / "yield" / n / "calls.jsonl").read_text().splitlines())}
               for n in JUDGES}

    def distinct(x, y):
        return sum(verdict[n].get((x, y)) is False and verdict[n].get((y, x)) is False for n in JUDGES) >= 2

    def yield_k(items, k):
        eligible = [x for x in items if x["ok"] and x["q"] >= k]
        for r in range(len(eligible), 0, -1):
            if any(all(distinct(a["id"], b["id"]) for a, b in itertools.combinations(c, 2))
                   for c in itertools.combinations(eligible, r)):
                return r
        return 0

    task_list = sorted(data)
    print(f"Mean over {len(task_list)} tasks; Y_k under the majority of {', '.join(JUDGES)}")
    print(f"{'method':12s} {'Q_max':>7s} {'Q_avg':>7s} " + " ".join(f"{'Y_' + str(k):>7s}" for k in ks))
    for m in methods:
        items = {t: data[t].get(m, []) for t in task_list}
        q_max = np.mean([max([x["q"] for x in items[t]] or [0.0]) for t in task_list])
        q_avg = np.mean([np.mean([x["q"] for x in items[t]]) if items[t] else 0.0 for t in task_list])
        ys = [np.mean([yield_k(items[t], k) for t in task_list]) for k in ks]
        print(f"{m:12s} {q_max:7.1f} {q_avg:7.1f} " + " ".join(f"{y:7.2f}" for y in ys))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("judge", "score"))
    parser.add_argument("--baselines-run", required=True, help="baselines run (it names its program QD run)")
    parser.add_argument("--judges", default=",".join(JUDGES))
    parser.add_argument("--ks", default="0,50,75,100", help="quality thresholds k for Y_k")
    args = parser.parse_args()
    run_dir = config.HERE / "runs" / args.baselines_run
    if args.stage == "judge":
        for name in args.judges.split(","):
            judge(name, run_dir)
    else:
        score(run_dir, [int(k) for k in args.ks.split(",")])


if __name__ == "__main__":
    main()
