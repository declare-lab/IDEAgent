"""Program-QD comparison baselines: stateless, single-shot (one-shot), sequential memory and closed-book NOVA, each run
by its own native runner under the same program-proposal condition as program QD.

    python -m verifiable.program_qd.baselines --name NAME --qd-run QD_RUN
    (run from the repository root; --stage resumes from a later stage; --frameworks a,b)

Matched to the QD run: the same topics file (task backgrounds with the baseline to beat), servers, ideator model and
replica per topic, sampling, reasoning effort, ideator budget, steno, transport guards, ideator system-prompt and
directive appendices, 10 proposals per topic, and one final run per proposal at the final budget. The native baseline
code is not modified. As in their native form, the baselines get no evaluator, critic, repair or execution feedback
during generation. After generation, every proposal's program gets the same code-correction loop as QD's drafts
(correct_programs: one run at the search budget, and a programming-bug crash gets up to config.CRASH_FIX_ATTEMPTS
single-turn ideator calls to fix the bug without changing the algorithm); then the freeze and the final runs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import yaml

sys.dont_write_bytecode = True

from verifiable import proposals  # noqa: E402
from verifiable.common import file_digest, load_json, now, write_json  # noqa: E402

from . import config, overlay, pipeline, prompts, tasks  # noqa: E402

# name: (native config, native runner, config section, module whose prompt globals the runner reads)
FRAMEWORKS = {
    "stateless": ("baselines/simple/stateless.yaml", "baselines/simple/run_simple_baseline.py", "baseline",
                  "baselines.simple.pipeline"),
    "single_shot": ("baselines/simple/single_shot.yaml", "baselines/simple/run_simple_baseline.py", "baseline",
                    "baselines.simple.pipeline"),
    "sequential": ("baselines/simple/sequential_memory.yaml", "baselines/simple/sequential_memory.py",
                   "sequential_memory", "ideagent.sequential_memory_baseline"),
    "nova": ("baselines/nova/nova_closed_book.yaml", "baselines/nova/run_nova_closed_book.py", "nova",
             "baselines.nova.pipeline"),
}
PROPOSALS_PER_TOPIC = 10   # stateless/single-shot/sequential n_ideas, NOVA final_k (native configs), QD free seeds
# Single-shot writes all 10 programs in one response: natively 8x the per-idea cap (65536 vs 8192); here 10 x 32k would
# exceed the 128k context, so it gets the context minus 16k for the prompt.
SINGLE_SHOT_MAX_NEW_TOKENS = 131072 - 16384
# Natively 3600 s. Stateless sends 10 requests per topic at once, so with every topic running, requests wait in the
# server queue; a client timeout would abort and resubmit them.
REQUEST_TIMEOUT_S = 6 * 3600
# NOVA's budget in the paper's verifiable runs: 2 rounds x 5 ideas (its open-ended config has 3 x 10).
NOVA_BUDGET = {"rounds": 2, "ideas_per_round": 5}
EMPTY_IDEA = "(no answer: the generator returned no content within its token budget, so there is no program)"
METHODS = ("qd",) + tuple(FRAMEWORKS)
children: list[subprocess.Popen] = []


# ── worker: one native baseline runner for one topic ─────────────────────────────────────
def install(framework: str, log_dir: Path) -> dict:
    """The program-proposal condition on the native baseline modules, plus QD's transport guards."""
    from ideagent import clients

    log_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def event(kind: str, **fields) -> None:
        with lock, (log_dir / "overlay_events.jsonl").open("a") as handle:
            handle.write(json.dumps({"at": time.time(), "event": kind, **fields}, default=str) + "\n")

    overlay.install_client_guards(event)
    from ideagent import signature

    module = importlib.import_module(FRAMEWORKS[framework][3])
    # The program-task prompts QD's ideator gets (prompts.py), read by the runner's module at call time; the directive
    # keeps the baselines' judgment-neutral memory.
    module.IDEATOR_SYSTEM_PROMPT = prompts.IDEATOR_SYSTEM
    module.op_free_baseline = prompts.op_free_baseline
    if framework == "single_shot":
        module.single_shot_directive = prompts.single_shot_directive
    if framework == "nova":
        module._REFLECT_SYSTEM, module._FINAL_SYSTEM = prompts.NOVA_REFLECT_SYSTEM, prompts.NOVA_FINAL_SYSTEM
    signature.SIGNATURE_SYSTEM = prompts.SIGNATURE_SYSTEM
    # No signature for an empty answer (the steno would invent one); the runners then use their native fallback.
    native_signature = module.extract_signature

    def extract_signature(idea_text, **kwargs):
        if not idea_text.strip() or idea_text.strip() == EMPTY_IDEA:
            return None
        return native_signature(idea_text, **kwargs)

    module.extract_signature = extract_signature
    # A guarded ideator call that ends with no content would natively abort the whole topic (min_idea_chars); here it
    # becomes one failed proposal (no program, performance 0), as a failed QD draft does.
    ideator_models = {spec["model"] for spec in config.IDEATOR_REPLICAS}
    native_generate = clients.VLLMOpenAIClient.generate

    def generate(self, messages, **kwargs):
        text = native_generate(self, messages, **kwargs)
        if self.model_id in ideator_models and kwargs.get("response_format") is None and not str(text).strip():
            event("empty_idea_placeholder", model=self.model_id)
            return EMPTY_IDEA
        return text

    clients.VLLMOpenAIClient.generate = generate
    directive = (module.single_shot_directive(PROPOSALS_PER_TOPIC) if framework == "single_shot"
                 else module.op_free_baseline([]))
    return {"framework": framework, "native_source_modified": False,
            "effective_prompts": {"ideator_system": module.IDEATOR_SYSTEM_PROMPT, "first_directive": directive},
            "changes": ["program-task ideator system prompt and directives (prompts.py, as in program QD); program-task "
                        "steno prompt and, for NOVA, selection prompts",
                        "QD's transport guards (resampling, fail closed, presence_penalty, reasoning_effort)",
                        "an empty ideator answer becomes a failed proposal instead of aborting the topic, with no "
                        "steno signature"]}


def use_suite(task_list) -> None:
    """The task suite's program contract, as in QD's overlay: its prompt wording and program language
    (ALE-Bench: C++); nothing changes for AlphaResearch."""
    suites = {tasks.suite(t) for t in task_list}
    if len(suites) == 1 and next(iter(suites)) in tasks.MODULES:
        prompts.adapt(tasks.MODULES[next(iter(suites))].PROMPT_EDITS)
    tasks.use_language(task_list)


def worker(framework: str, config_path: str, log_dir: Path) -> None:
    use_suite([Path(config_path).stem])          # configs/<framework>/<task>.yaml
    snapshot = install(framework, log_dir)
    snapshot["baselines_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (log_dir / "overlay_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    sys.argv = [str(config.ARCHIVE / FRAMEWORKS[framework][1]), "--config", config_path]
    runpy.run_path(sys.argv[0], run_name="__main__")


# ── prepare ──────────────────────────────────────────────────────────────────────────────
def framework_config(framework: str, run_dir: Path, task: str, index: int) -> dict:
    native_path, _, section, _ = FRAMEWORKS[framework]
    cfg = yaml.safe_load((config.ARCHIVE / native_path).read_text())
    cfg["data"] = {"input_jsonl": str(run_dir / "topics.jsonl"), "output_dir": str(run_dir / "generations" / framework / task)}
    # min_idea_chars 1: a short or failed answer is kept as a proposal (scored 0 if it has no valid program) instead of
    # aborting the topic.
    cfg[section].update(init_topic_idx=index, end_topic_idx=index + 1, resume=True, min_idea_chars=1)
    if framework == "nova":
        cfg[section].update(NOVA_BUDGET)
    cfg["client"] = {"request_timeout": REQUEST_TIMEOUT_S}
    cfg["generator"] = pipeline.role_agent("ideator", index)
    if framework == "single_shot":
        cfg["generator"]["max_new_tokens"] = SINGLE_SHOT_MAX_NEW_TOKENS
    cfg["steno"] = pipeline.role_agent("steno")
    if "judge" in cfg:
        cfg["judge"] = pipeline.role_agent("judge")   # NOVA's external selector
    return cfg


def prepare(run_dir: Path, qd_run: Path, frameworks: list[str], task_list: list[str]) -> dict:
    pipeline.snapshot_code(run_dir)
    qd = load_json(qd_run / "manifest.json")
    missing = [t for t in task_list if t not in qd["tasks"]]
    if missing:
        raise ValueError(f"Tasks not in the QD run: {missing}")
    shutil.copyfile(qd_run / "topics.jsonl", run_dir / "topics.jsonl")
    if file_digest(run_dir / "topics.jsonl") != qd["topics_sha256"]:
        raise RuntimeError("QD topics file changed since its run")
    configs = {}
    for framework in frameworks:
        for task in task_list:
            index = qd["tasks"].index(task)    # position in the topics file; also QD's ideator replica for the topic
            path = run_dir / "configs" / framework / f"{task}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(framework_config(framework, run_dir, task, index), sort_keys=False))
            configs[f"{framework}/{task}"] = {"path": str(path), "sha256": file_digest(path)}
    manifest = {
        "created_at": now(), "condition": f"{config.CONDITION}_baselines", "qd_run": str(qd_run),
        "frameworks": frameworks, "tasks": task_list, "topics": str(run_dir / "topics.jsonl"),
        "topics_sha256": qd["topics_sha256"], "baseline_scores": {t: qd["baseline_scores"][t] for t in task_list},
        "configs": configs, "proposals_per_topic": PROPOSALS_PER_TOPIC,
        "final_budget_s": config.FINAL_BUDGET_S, "single_shot_max_new_tokens": SINGLE_SHOT_MAX_NEW_TOKENS,
        "native_source_sha256": {p: file_digest(config.ARCHIVE / p) for p in sorted(
            {f[0] for f in FRAMEWORKS.values()} | {f[1] for f in FRAMEWORKS.values()}
            | {"baselines/simple/pipeline.py", "ideagent/sequential_memory_baseline.py",
               "baselines/nova/pipeline.py", "ideagent/prompts.py", "ideagent/agent_prompts.py"})},
        "program_qd_sha256": {p.name: file_digest(p) for p in sorted(config.HERE.glob("*.py"))},
        "roles": {"generator": pipeline.role_agent("ideator"), "steno": pipeline.role_agent("steno"),
                  "nova_selector": pipeline.role_agent("judge")},
        "serving": {"structured_outputs": config.STRUCTURED_OUTPUTS, "servers": list(pipeline.servers())},
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


# ── generation ───────────────────────────────────────────────────────────────────────────
def generate(run_dir: Path, manifest: dict, deadline: float) -> dict:
    processes = {}
    for key, entry in manifest["configs"].items():
        framework, task = key.split("/")
        logs = run_dir / "logs" / framework / task
        logs.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-m", "verifiable.program_qd.baselines", "--worker", framework,
                "--config", entry["path"], "--log-dir", str(logs)]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(config.ARCHIVE)}
        handle = (logs / "generation.log").open("ab")
        child = subprocess.Popen(args, cwd=config.ARCHIVE, env=env, stdout=handle, stderr=handle, start_new_session=True)
        children.append(child)
        processes[key] = child
    pipeline.log(run_dir, f"generation started: {len(processes)} processes ({', '.join(manifest['frameworks'])})")
    exit_codes = {}
    while len(exit_codes) < len(processes):
        for key, child in processes.items():
            if key not in exit_codes and child.poll() is not None:
                exit_codes[key] = child.returncode
                pipeline.log(run_dir, f"generation finished: {key} exit {child.returncode}")
        if time.time() > deadline:
            for key, child in processes.items():
                if key not in exit_codes:
                    os.killpg(child.pid, signal.SIGTERM)
                    exit_codes[key] = "deadline"
            break
        time.sleep(15)
    write_json(run_dir / "generation_completed.json", {"at": now(), "exit_codes": exit_codes})
    return exit_codes


# ── code correction (the loop QD's drafts get; config.CRASH_FIX_*) ───────────────────────────
CORRECTION_WORKERS = 16


def ask_fix(task: str, code: str, report: dict, event) -> str | None:
    """One single-turn bug-fix call, as in QD: the topic's ideator (model, replica and sampling) at the CRASH_FIX
    settings, given only the current program and its exception."""
    from ideagent.clients import build_client
    agent = pipeline.role_agent("ideator", config.TASKS.index(task) if task in config.TASKS else 0)
    client = build_client(model_id=agent["model_id"], vllm_port=agent["port"], api_key_env=agent["api_key_env"],
                          request_timeout=3600)          # one client per call: the guards read per-client metadata
    sampling = {k: agent[k] for k in ("temperature", "top_p", "top_k", "enable_thinking", "max_new_tokens") if k in agent}
    messages = [{"role": "system", "content": prompts.BUGFIX_SYSTEM},
                {"role": "user", "content": prompts.bugfix_user(code, report)}]
    try:
        text = client.generate(messages, **{**sampling, **config.CRASH_FIX_SETTINGS})
    except Exception as exc:      # a failed call is a failed attempt
        event("bug_fix_call_failed", task=task, error=f"{type(exc).__name__}: {exc}"[:300])
        return None
    return tasks.extract_program(text or "")


def correct_programs(run_dir: Path, manifest: dict, ask=ask_fix) -> dict:
    """Before the freeze, while the servers are up: run every generated proposal's program once at the search budget
    (as QD runs each draft) and give a programming-bug crash QD's correction loop (overlay.fix_program, same attempts,
    prompt and diff limits). A used fix replaces only the proposal's program block; the original files are kept as
    *.before_bugfix. A bug that survives every call leaves the proposal as generated, so its final run fails as before."""
    from concurrent.futures import ThreadPoolExecutor
    folder = run_dir / "bugfix"
    folder.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def event(kind: str, **fields) -> None:
        with lock, (folder / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"at": time.time(), "event": kind, **fields}, default=str) + "\n")

    overlay.install_client_guards(event)
    jobs = []
    for framework in manifest["frameworks"]:
        for task in manifest["tasks"]:
            for path in sorted((run_dir / "generations" / framework / task).glob("final_idea_cores*.jsonl")):
                for line_no, line in enumerate(path.read_text().splitlines()):
                    if not line.strip():
                        continue
                    for item_no, item in enumerate(json.loads(line).get("items") or []):
                        code = tasks.extract_program(item.get("idea_text") or "")
                        if code is not None:
                            jobs.append((framework, task, path, line_no, item_no, str(item.get("idea_id", item_no)), code))

    def one(job):
        framework, task, path, line_no, item_no, idea_id, code = job
        where = folder / framework / task / f"{path.stem}-{line_no}-{item_no}-{idea_id}"
        run = tasks.run_program(task, code, where, config.IDEATION_BUDGET_S)
        if not any(overlay.programming_bug(c) for c in run["cases"]):
            return job, None, None
        fixed, _, info = overlay.fix_program(
            task, code, run, where, config.IDEATION_BUDGET_S, lambda current, report: ask(task, current, report, event),
            attempts=config.CRASH_FIX_ATTEMPTS, max_changed=config.CRASH_FIX_MAX_CHANGED_LINES,
            max_total_changed=config.CRASH_FIX_MAX_TOTAL_CHANGED_LINES,
            log=lambda **fields: event("bug_fix", framework=framework, task=task, idea_id=idea_id, **fields))
        event("program_corrected", framework=framework, task=task, idea_id=idea_id,
              **{k: info[k] for k in ("attempts", "fixed", "unfixed", "changed_lines", "folder")})
        return job, (fixed if fixed != code else None), info

    with ThreadPoolExecutor(max_workers=CORRECTION_WORKERS) as pool:
        results = list(pool.map(one, jobs))
    fixes: dict = {}
    for (framework, task, path, line_no, item_no, idea_id, code), fixed, info in results:
        if fixed is not None:
            fixes.setdefault(path, []).append((line_no, item_no, fixed))
    for path, edits in fixes.items():
        shutil.copyfile(path, path.with_name(path.name + ".before_bugfix"))
        records = [json.loads(line) if line.strip() else None for line in path.read_text().splitlines()]
        for line_no, item_no, fixed in edits:
            item = records[line_no]["items"][item_no]
            item["idea_text"] = tasks.replace_program(item["idea_text"], fixed)
        path.write_text("".join(("" if r is None else json.dumps(r, ensure_ascii=False)) + "\n" for r in records))
    summary = {"programs_run": len(jobs),
               "bug_crashes": sum(info is not None for _, _, info in results),
               "fixed": sum(bool(info and info["fixed"]) for _, _, info in results),
               "unfixed": sum(bool(info and info["unfixed"]) for _, _, info in results),
               "per_framework": {f: {"bug_crashes": sum(1 for j, _, i in results if i is not None and j[0] == f),
                                     "fixed": sum(1 for j, _, i in results if i and i["fixed"] and j[0] == f)}
                                 for f in manifest["frameworks"]}}
    write_json(folder / "summary.json", {"at": now(), **summary})
    return summary


# ── freeze ───────────────────────────────────────────────────────────────────────────────
def freeze(run_dir: Path, manifest: dict) -> Path:
    """Every topic keeps its 10 slots: a topic whose native run failed gets an empty record, so its slots are frozen as
    failed proposals (performance 0) rather than dropped."""
    destination = run_dir / "frozen"
    if (destination / "proposals.json").exists():
        return destination / "proposals.json"
    sources = {}
    for framework in manifest["frameworks"]:
        paths = []
        for task in manifest["tasks"]:
            found = [p for p in sorted((run_dir / "generations" / framework / task).glob("final_idea_cores*.jsonl"))
                     if p.read_text().strip()]
            if not found:
                placeholder = run_dir / "failed_topics" / framework / f"{task}.jsonl"
                placeholder.parent.mkdir(parents=True, exist_ok=True)
                placeholder.write_text(json.dumps({"topic_id": task, "n_requested": PROPOSALS_PER_TOPIC, "items": [],
                                                   "generation_failed": f"see logs/{framework}/{task}/generation.log"}) + "\n")
                found = [placeholder]
            paths.extend(found)
        sources[framework] = paths
    registered = proposals.known_task
    proposals.known_task = lambda task: tasks.is_task(task) or registered(task)
    try:
        return Path(proposals.freeze(sources, destination)["bundle"])
    finally:
        proposals.known_task = registered


# ── report ───────────────────────────────────────────────────────────────────────────────
def method_results(run_dir: Path, qd_run: Path, frameworks: list[str]) -> dict:
    """{method: {task: [final result per frozen proposal]}}; QD from its own run."""
    out: dict = {}
    sources = [("qd", qd_run)] + [(f, run_dir) for f in frameworks]
    for method, folder in sources:
        bundle = folder / "frozen" / "proposals.json"
        finals_path = folder / "final_results.json"
        if not bundle.exists() or not finals_path.exists():
            continue
        finals = load_json(finals_path)["results"]
        for p in proposals.load_frozen(bundle)["proposals"]:
            if p["framework"] == method:
                out.setdefault(method, {}).setdefault(p["task"], []).append(finals.get(p["proposal_id"]) or {})
    return out


def report(run_dir: Path, manifest: dict) -> None:
    results = method_results(run_dir, Path(manifest["qd_run"]), manifest["frameworks"])
    methods = [m for m in METHODS if m in results]
    base = manifest["baseline_scores"]

    def perf(task, f):
        return tasks.performance(task, f["scores"], base[task]) if f.get("all_valid") else 0

    def best(task, method):
        values = [f["scores"] for f in results[method].get(task, []) if f.get("all_valid")]
        if not values:
            return None
        return max(values, key=lambda s: tasks.performance(task, s, base[task]))

    rows = ["| Task | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    wins = {m: 0 for m in methods}
    for task in manifest["tasks"]:
        top = {m: max((perf(task, f) for f in results[m].get(task, [])), default=0) for m in methods}
        lead = max(top.values())
        for m in methods:
            wins[m] += lead > 0 and top[m] == lead
        cells = []
        for m in methods:
            pick = best(task, m)
            score = "-" if pick is None else "/".join(f"{s:.6g}" for s in pick)
            mark = "**" if lead > 0 and top[m] == lead else ""
            cells.append(f"{mark}{score} ({top[m]}){mark}")
        rows.append(f"| {task} | " + " | ".join(cells) + " |")
    summary = ["| Method | Valid finals | Beat baseline | Mean performance (all slots) | Mean best performance per task "
               "| Tasks best (ties shared) |", "|---|---:|---:|---:|---:|---:|"]
    for m in methods:
        slots = [(t, f) for t in manifest["tasks"] for f in results[m].get(t, [])]
        valid = sum(bool(f.get("all_valid")) for _, f in slots)
        beat = sum(bool(f.get("all_valid")) and all(tasks.better(t, s, b) for s, b in zip(f["scores"], base[t]))
                   for t, f in slots)
        mean = sum(perf(t, f) for t, f in slots) / len(slots) if slots else 0
        best_mean = sum(max((perf(t, f) for f in results[m].get(t, [])), default=0)
                        for t in manifest["tasks"]) / len(manifest["tasks"])
        summary.append(f"| {m} | {valid}/{len(slots)} | {beat}/{len(slots)} | {mean:.1f} | {best_mean:.1f} | {wins[m]} |")
    text = "\n".join([
        "# Program QD vs baselines", "",
        f"QD run `{Path(manifest['qd_run']).name}`, baselines run `{run_dir.name}`. Same task backgrounds, models, "
        f"sampling and program-proposal prompts; {PROPOSALS_PER_TOPIC} frozen proposals per topic per method, each "
        f"program run once at {config.FINAL_BUDGET_S} s. Performance: 0 invalid or missing, 20 valid with no gain over "
        "the baseline, 100 at the best-known value (uncapped). Only QD runs programs and uses evaluators and critics "
        "during the search; every method's programs get the same code-correction loop for programming bugs.", "",
        *summary, "",
        "Best final score per task (instance scores joined by /) and its performance; bold = best method on the task.", "",
        *rows])
    (run_dir / "report.md").write_text(text + "\n")


# ── main ─────────────────────────────────────────────────────────────────────────────────
STAGES = ("prepare", "generation", "freeze", "final", "report")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--qd-run", help="name of the program QD run to match and compare against")
    parser.add_argument("--frameworks", default=",".join(FRAMEWORKS))
    parser.add_argument("--tasks", default="", help="default: every task of the QD run")
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--stage", choices=STAGES, default="prepare", help="first stage to run (resume)")
    parser.add_argument("--external-servers", action="store_true",
                        help="use servers already running (pipeline --serve) instead of starting and stopping them")
    parser.add_argument("--worker", choices=tuple(FRAMEWORKS), help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--log-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.config, Path(args.log_dir))
        return
    run_dir = config.HERE / "runs" / args.name
    qd_run = config.HERE / "runs" / args.qd_run
    deadline = time.time() + args.hours * 3600
    start = STAGES.index(args.stage)
    run_dir.mkdir(parents=True, exist_ok=start > 0)

    def terminate(signum, frame):
        pipeline.log(run_dir, f"terminated by signal {signum}")
        for child in reversed(children + pipeline.children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        os._exit(124)

    signal.signal(signal.SIGTERM, terminate)
    try:
        if start == 0:
            task_list = args.tasks.split(",") if args.tasks else load_json(qd_run / "manifest.json")["tasks"]
            manifest = prepare(run_dir, qd_run, args.frameworks.split(","), task_list)
            pipeline.log(run_dir, f"prepared {len(manifest['configs'])} framework/topic configs")
        else:
            manifest = load_json(run_dir / "manifest.json")
        suites = {tasks.suite(t) for t in manifest["tasks"]}
        if len(suites) > 1 and suites & set(tasks.MODULES):
            raise ValueError(f"One task suite per baselines run (program contract and language): {sorted(suites)}")
        if start <= 1:
            if args.external_servers:
                pipeline.check_external(run_dir)
            servers = [] if args.external_servers else pipeline.launch_all(run_dir)
            try:
                check = pipeline.preflight(run_dir)
                pipeline.log(run_dir, f"preflight ok={check['ok']}: {json.dumps(check)[:600]}")
                if not check["ok"]:
                    raise RuntimeError("Preflight failed; see preflight/")
                use_suite(manifest["tasks"])      # after the preflight (its sample program is Python)
                exit_codes = generate(run_dir, manifest, deadline)
                if config.CRASH_FIX_ATTEMPTS > 0:
                    try:
                        summary = correct_programs(run_dir, manifest)
                        pipeline.log(run_dir, f"code correction: {json.dumps(summary)}")
                    except Exception as exc:     # never lose the baselines over it: freeze the proposals as generated
                        pipeline.log(run_dir, f"WARNING: code correction failed ({type(exc).__name__}: {exc}); "
                                              "proposals frozen as generated")
            finally:
                for server in servers:
                    pipeline.stop(server)
            if any(code != 0 for code in exit_codes.values()):
                pipeline.log(run_dir, f"WARNING: some generation processes did not exit cleanly: {exit_codes}")
        use_suite(manifest["tasks"])
        bundle = freeze(run_dir, manifest) if start <= 2 else run_dir / "frozen" / "proposals.json"
        pipeline.log(run_dir, f"frozen {proposals.load_frozen(bundle)['proposal_count']} proposals")
        finals = pipeline.final_runs(run_dir, bundle) if start <= 3 else load_json(run_dir / "final_results.json")["results"]
        pipeline.log(run_dir, f"final runs done: {sum(bool(f.get('all_valid')) for f in finals.values())}/{len(finals)} valid")
        report(run_dir, manifest)
        pipeline.log(run_dir, "completed")
    except BaseException as exc:
        write_json(run_dir / f"failed-{int(time.time())}.json", {"at": now(), "reason": f"{type(exc).__name__}: {exc}",
                                                                 "traceback": traceback.format_exc()})
        pipeline.log(run_dir, f"FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        for child in reversed(children + pipeline.children):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
