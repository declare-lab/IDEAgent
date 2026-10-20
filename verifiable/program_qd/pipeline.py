"""Program QD end to end: prepare -> ideation (native QD + overlay) -> freeze -> final runs -> report.

    python -m verifiable.program_qd.pipeline --name NAME [--tasks a,b]
    (run from the repository root; --stage resumes from a later stage)

No converter: each frozen proposal's own program is run once at the final budget.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

import yaml

sys.dont_write_bytecode = True

from verifiable import proposals  # noqa: E402
from verifiable.common import file_digest, load_json, now, write_json  # noqa: E402

from . import config, overlay, tasks  # noqa: E402

ROLES = ("ideator", "critic", "quality", "steno", "judge")
ACCEPTED = {"accepted", "replaced", "keep_accepted"}
STAGES = ("prepare", "ideation", "freeze", "final", "report")
children: list[subprocess.Popen] = []


def log(run_dir: Path, message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with (run_dir / "run.log").open("a") as handle:
        handle.write(line + "\n")


def role_agent(role: str, topic_index: int = 0) -> dict:
    """Native agent config: the ideator on its replica for this topic; every other role on the evaluator."""
    settings = config.ROLE_SETTINGS[role]
    if role == "ideator":
        server, sampling = config.ideator_server(topic_index), config.QWEN38_THINKING
    else:
        server = config.EVALUATOR
        sampling = config.QWEN36_THINKING if settings["enable_thinking"] else config.QWEN36_NON_THINKING
    return {"model_id": server["model"], "api_key_env": "VLLM_API_KEY", "port": server["port"], **sampling,
            "enable_thinking": settings["enable_thinking"], "max_new_tokens": settings["max_new_tokens"]}


def servers() -> tuple[dict, ...]:
    return config.SERVERS


# ── prepare ──────────────────────────────────────────────────────────────────────────────
def baseline_scores(run_dir: Path, task: str) -> list[float]:
    run = tasks.run_program(task, tasks.BASELINE_CODE[task], run_dir / "baselines" / task, config.IDEATION_BUDGET_S)
    if not run["all_valid"]:
        raise RuntimeError(f"Baseline invalid for {task}: {run['cases']}")
    return run["scores"]


def snapshot_code(run_dir: Path) -> None:
    """The exact program_qd code this run uses."""
    folder = run_dir / "code_snapshot"
    folder.mkdir(parents=True, exist_ok=True)
    for path in config.HERE.glob("*.py"):
        (folder / path.name).write_bytes(path.read_bytes())


def prepare(run_dir: Path, task_list: list[str]) -> dict:
    snapshot_code(run_dir)
    unknown = [t for t in task_list if not tasks.is_task(t)]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    baselines = {task: baseline_scores(run_dir, task) for task in task_list}
    topics = []
    for task in task_list:
        text = tasks.background(task, baselines[task], config.IDEATION_BUDGET_S, config.FINAL_BUDGET_S)
        spec = {"paper_id": f"spec:{task}:program:v1", "title": f"Task: {tasks.TASKS[task]['title']} (not a published paper)",
                "abstract": text, "full_text": text,
                "provenance": {"kind": "task_specification", "source": tasks.source_url(task)}}
        topics.append({"topic_id": task, "background_papers": [spec], "target_papers": [spec], "verification_task": task,
                       "target_record_purpose": "Loader compatibility only; same explicit specification."})
    topic_path = run_dir / "topics.jsonl"
    topic_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in topics))
    from ideagent.data import load_raw_topic_groups
    load_raw_topic_groups(topic_path)  # the native loader accepts the topics
    native = yaml.safe_load(config.NATIVE_QD_CONFIG.read_text())
    configs = {}
    for index, task in enumerate(task_list):
        cfg = json.loads(json.dumps(native))
        cfg["data"] = {"input_jsonl": str(topic_path), "output_dir": str(run_dir / "generations" / task)}
        cfg["run"].update(init_topic_idx=index, end_topic_idx=index + 1, resume=True,
                          press_feasibility_target=config.PRESS_FEASIBILITY_TARGET,
                          soundness_n=config.SOUNDNESS_JUDGMENTS, soundness_fatal_min_votes=1)
        cfg["client"] = {"request_timeout": 3600}
        cfg["agents"] = {role: role_agent(role, index) for role in ROLES}
        path = run_dir / "configs" / f"qd_{task}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[task] = {"path": str(path), "sha256": file_digest(path), "ideator_server": config.ideator_server(index)["name"]}
    manifest = {
        "created_at": now(), "condition": config.CONDITION, "tasks": task_list, "topics": str(topic_path),
        "topics_sha256": file_digest(topic_path), "configs": configs, "baseline_scores": baselines,
        "budgets_s": {"ideation_per_program": config.IDEATION_BUDGET_S, "final_per_program": config.FINAL_BUDGET_S},
        "quality": {"runtime_tiebreak": config.RUNTIME_TIEBREAK,
                    "q": "performance + nb_tiebreak * non-obviousness - runtime_tiebreak * total solve seconds",
                    "nb_tiebreak": config.NB_TIEBREAK,
                    "performance": "0 invalid/no program; 20 valid with no progress over the baseline; 20 + 80 x mean "
                                   "progress from the baseline to the best-known value (internal, not shown); uncapped",
                    "acceptance_gate": "valid program beating the baseline on every instance, soundness panel not fatal "
                                       "(one judgment: the program implements the proposed method and both are meant "
                                       "for this task), native diversity gate; NB, clarity and the soundness mean do "
                                       "not gate; a non-fatal miss is repairable",
                    "refinement": "every accepted idea is refined further within native limits",
                    "references_internal": {t: tasks.TASKS[t]["reference"] for t in task_list}},
        "native_run_section_changes": {"press_feasibility_target": config.PRESS_FEASIBILITY_TARGET,
                                       "soundness_n": config.SOUNDNESS_JUDGMENTS, "soundness_fatal_min_votes": 1},
        "v7": {"freeze": "one slot per free seed: the best-scoring version among that seed's drafts"},
        "native_source_sha256": {str(p.relative_to(config.ARCHIVE)): file_digest(p) for p in sorted(
            list((config.ARCHIVE / "ideagent").glob("*.py")) + [config.ARCHIVE / "scripts/run_ideagent.py",
                                                                         config.NATIVE_QD_CONFIG])},
        "program_qd_sha256": {p.name: file_digest(p) for p in sorted(config.HERE.glob("*.py"))},
        "task_sources": {t: {"url": tasks.source_url(t), "files_sha256": tasks.source_files(t),
                             "stricter_than_upstream": tasks.STRICTER.get(t)} for t in task_list},
        "roles": {role: role_agent(role) for role in ROLES},
        "presence_penalty": config.PRESENCE_PENALTY,
        "serving": {"structured_outputs": config.STRUCTURED_OUTPUTS, "servers": list(servers())},
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


# ── serving ──────────────────────────────────────────────────────────────────────────────
def launch(spec: dict, run_dir: Path) -> subprocess.Popen:
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", spec["port"])) == 0:
            raise RuntimeError(f"Port {spec['port']} occupied; refusing to adopt an unknown server")
    args = [config.VPY, "-m", "vllm.entrypoints.openai.api_server", "--model", spec["model"],
            *(["--revision", spec["revision"]] if spec.get("revision") else []), "--served-model-name", spec["model"],
            "--host", "127.0.0.1", "--port", str(spec["port"]), "--tensor-parallel-size", str(spec.get("tp", 1)),
            "--dtype", "bfloat16", "--gpu-memory-utilization", str(spec["memory"]), "--max-num-seqs",
            str(spec["max_num_seqs"]), "--max-num-batched-tokens", str(spec["max_num_batched_tokens"]),
            "--enable-prefix-caching", "--enable-chunked-prefill", "--language-model-only", "--generation-config", "vllm",
            "--structured-outputs-config", config.STRUCTURED_OUTPUTS]
    args += ["--max-model-len", str(spec["max_model_len"]),
             "--reasoning-parser", spec.get("reasoning_parser", "qwen3")]                # Qwen: thinking by default
    if spec.get("dp", 1) > 1:
        args += ["--data-parallel-size", str(spec["dp"])]
    if spec.get("speculative"):
        args += ["--speculative-config", spec["speculative"]]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": spec["gpu"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "8",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
           "PATH": str(Path(config.VPY).parent) + os.pathsep + os.environ.get("PATH", "")}
    folder = run_dir / "servers" / spec["name"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "vllm.log").open("ab") as handle:
        child = subprocess.Popen(args, cwd=config.ARCHIVE, env=env, stdout=handle, stderr=handle, start_new_session=True)
    children.append(child)
    write_json(folder / f"started-{int(time.time())}.json", {"at": now(), "pid": child.pid, "argv": args})
    until = time.monotonic() + 1200
    while True:
        if child.poll() is not None:
            raise RuntimeError(f"{spec['name']} exited during startup ({child.returncode}); see {folder}/vllm.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{spec['port']}/v1/models", timeout=3) as response:
                if spec["model"] in [m["id"] for m in json.load(response)["data"]]:
                    break
        except Exception:
            pass
        if time.monotonic() > until:
            raise TimeoutError(f"{spec['name']} not ready after 1200 s")
        time.sleep(3)
    log(run_dir, f"{spec['name']} ready on port {spec['port']}")
    return child


def stop(child: subprocess.Popen) -> None:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)


def check_external(run_dir: Path) -> None:
    """--external-servers: every configured server must already be up (started with --serve) with its model."""
    for spec in servers():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{spec['port']}/v1/models", timeout=10) as response:
                models = [m["id"] for m in json.load(response)["data"]]
        except Exception as exc:
            raise RuntimeError(f"{spec['name']} is not serving on port {spec['port']}: {exc}") from exc
        if spec["model"] not in models:
            raise RuntimeError(f"Port {spec['port']} serves {models}, not {spec['model']}")
        log(run_dir, f"{spec['name']} external on port {spec['port']}")


def launch_all(run_dir: Path, specs=None) -> list[subprocess.Popen]:
    """GPUs start in parallel; servers sharing a GPU start one after another (vLLM's memory check must not overlap)."""
    specs = specs or servers()
    by_gpu: dict = {}
    for spec in specs:
        by_gpu.setdefault(spec["gpu"], []).append(spec)
    started: list[subprocess.Popen] = []

    def bring_up(group):
        for spec in group:
            started.append(launch(spec, run_dir))

    with ThreadPoolExecutor(max_workers=len(by_gpu)) as pool:
        futures = [pool.submit(bring_up, group) for group in by_gpu.values()]
    errors = [f.exception() for f in futures if f.exception()]
    if errors:
        for server in started:
            stop(server)
        raise errors[0]
    return started


# ── preflight ────────────────────────────────────────────────────────────────────────────
SAMPLE_PROPOSAL = """Title: Conway-set embedding for MSTD.
Insight: A = {0,2,3,4,7,11,12,14} has |A+A| = 26 and |A-A| = 25.
Mechanism: return the indicator vector of this set inside {0, ..., N-1}.
```python
S = [0, 2, 3, 4, 7, 11, 12, 14]

def solve(problem):
    return {"indicators": [1 if i in S else 0 for i in range(problem["N"])]}
```"""


def preflight(run_dir: Path) -> dict:
    """Live checks with the same transport guards the overlay installs (removed afterwards)."""
    from ideagent import clients as native_clients
    from ideagent.agents import QualityAgent, evaluate_soundness_multi
    from ideagent.clients import build_client
    from ideagent.diversity_judge import _response_format
    from ideagent.generation_loop import _score_map

    folder = run_dir / "preflight"
    folder.mkdir(parents=True, exist_ok=True)
    events = folder / "events.jsonl"

    def event(kind, **fields):
        with events.open("a") as handle:
            handle.write(json.dumps({"at": time.time(), "event": kind, **fields}, default=str) + "\n")

    original = native_clients.VLLMOpenAIClient.generate_many
    overlay.install_client_guards(event)
    try:
        report = {"at": now()}
        judge_messages = [
            {"role": "system", "content": "You are a strict novelty judge. Return the structured verdict."},
            {"role": "user", "content": "CANDIDATE:\n[i003] problem: MSTD sets | mechanism: Conway-set embedding\n\n"
                                        "ACCEPTED IDEAS (archive):\n[i001] problem: MSTD sets | mechanism: random "
                                        "greedy sum-set growth\n\nHISTORICAL QUALIFIED MECHANISMS:\n(none yet)\n\n"
                                        "REJECTED MEMORY:\n(none yet)\n\nReturn the structured novelty verdict."}]
        judge = []
        evaluator = build_client(model_id=config.EVALUATOR["model"], vllm_port=config.EVALUATOR["port"],
                                 api_key_env="VLLM_API_KEY", request_timeout=1200)
        kwargs = {**config.QWEN36_THINKING, **config.ROLE_SETTINGS["judge"], "response_format": _response_format(),
                  "presence_penalty": overlay.presence_penalty(config.EVALUATOR["model"], config.QWEN36_THINKING["temperature"])}
        raw = evaluator._chat_completion_create(**evaluator._chat_request(judge_messages, kwargs)).model_dump(mode="json")
        message = raw["choices"][0]["message"]
        judge.append({"server": config.EVALUATOR["name"],
                      "reasoning_chars": len(message.get("reasoning_content") or message.get("reasoning") or ""),
                      "guarded_parsed": overlay.json_ok(evaluator.generate(judge_messages, **kwargs))})
        report["judge"] = judge
        report["ideator"] = []
        for spec in config.IDEATOR_REPLICAS:
            ideator = build_client(model_id=spec["model"], vllm_port=spec["port"], api_key_env="VLLM_API_KEY",
                                   request_timeout=1200)
            raw = ideator._chat_completion_create(**ideator._chat_request([{"role": "user", "content":
                "Write a ```python block defining solve(problem) that returns {'indicators': [1] * problem['N']}."}],
                {**config.QWEN38_THINKING, "enable_thinking": True, "max_new_tokens": 8192})
            ).model_dump(mode="json")
            message = raw["choices"][0]["message"]
            report["ideator"].append({"server": spec["name"],
                                      "reasoning_chars": len(message.get("reasoning_content") or message.get("reasoning") or ""),
                                      "program_found": tasks.extract_program(message.get("content") or "") is not None})
        feedback_kwargs = {**config.QWEN36_THINKING, **config.ROLE_SETTINGS["quality"]}
        agent = QualityAgent(role="quality", client=evaluator, gen_kwargs=feedback_kwargs, single_turn=True)
        agent.reset_for_topic(prior_final_idea_cores=[])
        scores = _score_map(agent.evaluate(prev_critic_turn="", latest_ideator_turn=SAMPLE_PROPOSAL).get("rubric_assessment") or [])
        panel = evaluate_soundness_multi(SAMPLE_PROPOSAL, client=evaluator, gen_kwargs=feedback_kwargs,
                                         n=config.SOUNDNESS_JUDGMENTS)
        run = tasks.run_program("ar_mstd", tasks.extract_program(SAMPLE_PROPOSAL), folder / "program", config.IDEATION_BUDGET_S)
        report.update(rubric=scores, panel_samples=len(panel), program_scores=run["scores"],
                      transport_events=[json.loads(line)["event"] for line in events.read_text().splitlines()] if events.exists() else [])
        report["ok"] = (all(j["reasoning_chars"] > 0 and j["guarded_parsed"] for j in judge)
                        and all(i["reasoning_chars"] > 0 and i["program_found"] for i in report["ideator"])
                        and {"non_obviousness", "mechanism_clarity"} <= set(scores) and len(panel) == config.SOUNDNESS_JUDGMENTS
                        and run["scores"] == [1.04])
    finally:
        native_clients.VLLMOpenAIClient.generate_many = original
    write_json(folder / "preflight.json", report)
    return report


# ── ideation ─────────────────────────────────────────────────────────────────────────────
def generate(run_dir: Path, manifest: dict, deadline: float) -> dict:
    processes = {}
    for task in manifest["tasks"]:
        logs = run_dir / "logs" / task
        logs.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, str(config.HERE / "overlay.py"), "--config", manifest["configs"][task]["path"],
                "--log-dir", str(logs), "--run-dir", str(run_dir), "--task", task,
                "--baseline", json.dumps(manifest["baseline_scores"][task]),
                "--budget-s", str(config.IDEATION_BUDGET_S)]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(config.ARCHIVE)}
        handle = (logs / "generation.log").open("ab")
        child = subprocess.Popen(args, cwd=config.ARCHIVE, env=env, stdout=handle, stderr=handle, start_new_session=True)
        children.append(child)
        processes[task] = child
        log(run_dir, f"ideation started: {task} (pid {child.pid})")
    exit_codes = {}
    while len(exit_codes) < len(processes):
        for task, child in processes.items():
            if task not in exit_codes and child.poll() is not None:
                exit_codes[task] = child.returncode
                log(run_dir, f"ideation finished: {task} exit {child.returncode}")
        if time.time() > deadline:
            for task, child in processes.items():
                if task not in exit_codes:
                    os.killpg(child.pid, signal.SIGTERM)
                    exit_codes[task] = "deadline"
            break
        time.sleep(15)
    write_json(run_dir / "ideation_completed.json", {"at": now(), "exit_codes": exit_codes})
    return exit_codes


SLOTS = 10   # frozen proposals per task, as for every baseline


def v7_final_file(run_dir: Path, task: str) -> Path:
    """The task's frozen slots, written in the native final-file format: one per free seed, the best-scoring version
    (performance, then non-obviousness) among that seed's drafts in the candidates log (natively the slot holds the
    seed's current representative, which can be below its best draft)."""
    folder = run_dir / "generations" / task
    rows = [json.loads(line) for line in (folder / f"candidates_{task}.jsonl").read_text().splitlines() if line.strip()]
    best: dict = {}
    for row in rows:
        key = (row["feasibility"], row.get("non_obviousness") or 0)
        if row["seed_index"] not in best or key > best[row["seed_index"]][0]:
            best[row["seed_index"]] = (key, row)
    items = [{"idea_id": row["id"], "idea_text": row["idea_text"], "final_idea_core": row.get("signature") or {},
              "source_metadata": {"final_internal_outcome": row["outcome"], "lineage_id": row["lineage_id"],
                                  "seed_index": seed, "operator": row["operator"],
                                  "lineage_selection_policy": "v7: the best-scoring version among the seed's drafts"}}
             for seed, (_, row) in sorted(best.items())][:SLOTS]
    path = folder / "final_idea_cores_v7.jsonl"
    path.write_text(json.dumps({"topic_id": task, "free_seed_budget": SLOTS, "items": items}, ensure_ascii=False) + "\n")
    return path


def freeze(run_dir: Path, manifest: dict) -> Path:
    destination = run_dir / "frozen"
    if (destination / "proposals.json").exists():
        return destination / "proposals.json"
    paths = []
    for task in manifest["tasks"]:
        native = [p for p in (run_dir / "generations" / task).glob("final_idea_cores*.jsonl")   # _from<k> or none
                  if p.name != "final_idea_cores_v7.jsonl"]
        if not native:
            raise RuntimeError(f"No native final file for {task}; ideation incomplete")
        paths.append(v7_final_file(run_dir, task))
    # The shared freeze accepts registry tasks only; widen the check in this process (the registry is not edited).
    registered = proposals.known_task
    proposals.known_task = lambda task: tasks.is_task(task) or registered(task)
    try:
        return Path(proposals.freeze({"qd": paths}, destination)["bundle"])
    finally:
        proposals.known_task = registered


# ── final runs ───────────────────────────────────────────────────────────────────────────
def final_one(run_dir: Path, proposal: dict) -> dict:
    """One run per frozen proposal at the final budget. The slot is reserved first, so a randomized program cannot be
    re-run to pick a lucky outcome; a slot killed before its result gets one flagged rerun."""
    pid, task = proposal["proposal_id"], proposal["task"]
    slot = run_dir / "final" / pid
    if (slot / "result.json").exists():
        return load_json(slot / "result.json")
    code = tasks.extract_program(proposal["idea_text"])
    rerun = slot.exists()
    if rerun:
        slot = run_dir / "final_reruns" / pid
    slot.mkdir(parents=True)
    write_json(slot / "started.json", {"at": now(), "idea_sha256": proposal["idea_sha256"], "budget_s": config.FINAL_BUDGET_S})
    if code is None:
        result = {"task": task, "program_found": False, "all_valid": False, "scores": None}
    else:
        result = {**tasks.run_program(task, code, slot, config.FINAL_BUDGET_S), "program_found": True}
    result.update(proposal_id=pid, rerun_after_interrupt=rerun)
    write_json(slot / "result.json", result)
    return result


def final_runs(run_dir: Path, bundle: Path) -> dict:
    """With PQD_FINAL_LOCK set (concurrent runs), one run's final runs at a time hold that file lock, so concurrent runs
    never overlap their CPU-bound 300 s final runs."""
    frozen = proposals.load_frozen(bundle)["proposals"]
    lock = os.environ.get("PQD_FINAL_LOCK")
    handle = open(lock, "a") if lock else None
    if handle:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX)
        log(run_dir, f"final-run lock acquired ({lock})")
    try:
        # AlphaResearch programs are single-threaded: FINAL_WORKERS at once. ALE-Bench programs already run their
        # instances in parallel (alebench.PARALLEL), so one program at a time:
        # running many at once oversubscribes the CPUs and turns near-limit programs into timeouts.
        workers = config.FINAL_WORKERS if all(tasks.suite(p["task"]) not in tasks.MODULES for p in frozen) else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = dict(zip([p["proposal_id"] for p in frozen], pool.map(lambda p: final_one(run_dir, p), frozen)))
    finally:
        if handle:
            handle.close()
    write_json(run_dir / "final_results.json", {"at": now(), "budget_s": config.FINAL_BUDGET_S, "results": results})
    return results


# ── report ───────────────────────────────────────────────────────────────────────────────
def gate_reasons(run_dir: Path, task: str, idea_id: str) -> str:
    """Failing acceptance gates from the logged gate_results (the gates the overlay applies)."""
    path = run_dir / "generations" / task / f"candidates_{task}.jsonl"
    record = None
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("id") == idea_id:
                record = row
    if record is None:
        return "unknown"
    g = record.get("gate_results") or {}
    if g.get("full_acceptance_gate_pass"):
        return "passed"
    if not (record.get("idea_text") or "").strip():
        return "empty draft"
    failed = [label for key, label in (("eval_ok", "eval_fail"), ("baseline_pass", "baseline"),
                                       ("soundness_not_fatal", "fatal")) if not g.get(key)]
    if not g.get("not_duplicate", True):
        failed.append("duplicate")
    elif not g.get("diversity_pass", True):
        failed.append("diversity")
    return "+".join(failed) or "other"


def draft_runs(run_dir: Path, task: str) -> list[dict]:
    path = run_dir / "logs" / task / "overlay_events.jsonl"
    if not path.exists():
        return []
    return [e for e in map(json.loads, path.read_text().splitlines()) if e["event"] == "program_run"]


def report(run_dir: Path, manifest: dict, bundle: Path, finals: dict) -> None:
    frozen = proposals.load_frozen(bundle)["proposals"]
    rows = ["| Task | QD accepted | Drafts run valid | Final: program found / valid | Final: beat baseline | "
            "Best final score per instance (baseline; best known) | Beat best known | Mean runtime of valid finals (s) |",
            "|---|---:|---:|---:|---:|---|---:|---:|"]
    reasons_rows = ["| Task | Rejected representatives by failing gates |", "|---|---|"]
    for task in manifest["tasks"]:
        items = [p for p in frozen if p["task"] == task]
        base, best_known = manifest["baseline_scores"][task], tasks.TASKS[task]["reference"]
        drafts = draft_runs(run_dir, task)
        fin = [finals.get(p["proposal_id"]) or {} for p in items]
        valid = [f for f in fin if f.get("all_valid")]
        beat_base = sum(all(tasks.better(task, s, b) for s, b in zip(f["scores"], base)) for f in valid)
        beat_best = sum(all(tasks.better(task, s, r) for s, r in zip(f["scores"], best_known)) for f in valid)
        top = []
        for i, (b, r) in enumerate(zip(base, best_known)):
            values = [f["scores"][i] for f in valid]
            pick = (max if tasks.TASKS[task]["direction"] == "max" else min)(values) if values else None
            top.append(f"{'-' if pick is None else f'{pick:.6g}'} ({b:.6g}; {r:.6g})")
        runtimes = [c["runtime_s"] for f in valid for c in f["cases"] if c.get("runtime_s") is not None]
        outcomes = [p["original_item"]["source_metadata"].get("final_internal_outcome") for p in items]
        rows.append(f"| {task} | {sum(o in ACCEPTED for o in outcomes)}/{len(items)} | "
                    f"{sum(d['all_valid'] for d in drafts)}/{len(drafts)} | {sum(bool(f.get('program_found')) for f in fin)}/"
                    f"{len(valid)} | {beat_base} | {'; '.join(top)} | {beat_best} | "
                    f"{(sum(runtimes) / len(runtimes)) if runtimes else 0:.1f} |")
        reasons: dict = {}
        for p, outcome in zip(items, outcomes):
            if outcome not in ACCEPTED:
                key = gate_reasons(run_dir, task, p["original_item"]["idea_id"])
                reasons[key] = reasons.get(key, 0) + 1
        reasons_rows.append(f"| {task} | {', '.join(f'{k}: {v}' for k, v in sorted(reasons.items())) or '-'} |")
    reruns = sorted(pid for pid, f in finals.items() if f.get("rerun_after_interrupt"))
    text = "\n".join([
        "# Program QD run", "",
        f"Run `{run_dir.name}`. Native QD where each proposal contains its complete program. Every draft's program is "
        f"run during the search ({config.IDEATION_BUDGET_S} s per program); Q = performance, ties by "
        "non-obviousness, then runtime. Each frozen "
        f"representative's program is then run once at {config.FINAL_BUDGET_S} s per program (AlphaResearch's budget). "
        "Every frozen representative is in the denominator. Best-known values are shown here only; the ideator saw "
        "only the baseline.", "", *rows, "",
        *(["Final runs repeated once after an external interruption: " + ", ".join(reruns), ""] if reruns else []),
        "## Why frozen representatives were rejected", "", *reasons_rows, "",
        "Drafts run valid: drafts whose program produced a valid construction on every instance during the search."])
    (run_dir / "report.md").write_text(text + "\n")


# ── main ─────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--tasks", default=",".join(config.TASKS))
    parser.add_argument("--hours", type=float, default=14.0)
    parser.add_argument("--stage", choices=STAGES, default="prepare", help="first stage to run (resume)")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--serve", action="store_true",
                        help="only start the model servers and keep them up until this process is stopped")
    parser.add_argument("--external-servers", action="store_true",
                        help="use servers already running (from --serve) instead of starting and stopping them")
    args = parser.parse_args()
    run_dir = config.HERE / "runs" / args.name
    deadline = time.time() + args.hours * 3600
    start = STAGES.index(args.stage)
    run_dir.mkdir(parents=True, exist_ok=start > 0)

    def terminate(signum, frame):
        log(run_dir, f"terminated by signal {signum}")
        for child in reversed(children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        os._exit(124)

    signal.signal(signal.SIGTERM, terminate)
    if args.serve:
        launch_all(run_dir)
        log(run_dir, "serving until stopped (SIGTERM stops the servers)")
        while True:
            time.sleep(3600)
    try:
        task_list = list(config.SUITES.get(args.tasks) or args.tasks.split(","))
        manifest = prepare(run_dir, task_list) if start == 0 else load_json(run_dir / "manifest.json")
        if start == 0:
            log(run_dir, f"prepared; baselines {json.dumps(manifest['baseline_scores'])}")
        if start <= 1:
            if args.external_servers:
                check_external(run_dir)
            servers = [] if args.external_servers else launch_all(run_dir)
            try:
                check = preflight(run_dir)
                log(run_dir, f"preflight ok={check['ok']}: {json.dumps(check)[:1200]}")
                if not check["ok"]:
                    raise RuntimeError("Preflight failed; see preflight/")
                if args.preflight_only:
                    return
                tasks.use_language(manifest["tasks"])     # after the preflight (its sample program is Python)
                exit_codes = generate(run_dir, manifest, deadline)
            finally:
                for server in servers:
                    stop(server)
            if any(code != 0 for code in exit_codes.values()):
                log(run_dir, f"WARNING: some ideation processes did not exit cleanly: {exit_codes}")
        tasks.use_language(manifest["tasks"])
        bundle = freeze(run_dir, manifest) if start <= 2 else run_dir / "frozen" / "proposals.json"
        log(run_dir, f"frozen {proposals.load_frozen(bundle)['proposal_count']} proposals")
        finals = final_runs(run_dir, bundle) if start <= 3 else load_json(run_dir / "final_results.json")["results"]
        log(run_dir, f"final runs done: {sum(bool(f.get('all_valid')) for f in finals.values())}/{len(finals)} valid")
        report(run_dir, manifest, bundle, finals)
        log(run_dir, "completed")
    except BaseException as exc:
        write_json(run_dir / f"failed-{int(time.time())}.json", {"at": now(), "reason": f"{type(exc).__name__}: {exc}",
                                                                 "traceback": traceback.format_exc()})
        log(run_dir, f"FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
