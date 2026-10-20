"""Runtime overlay that runs the IDEAgent search on program tasks; the ideagent package itself is not modified.

install() patches, for one task:
  1. prompts: the program-task prompts (prompts.py) replace the native ideator, directive, rubric, soundness, critic,
     relative-NB, diversity-judge, consolidation and steno prompts; every evaluator and critic sees the task background;
  2. transport guards (install_client_guards): an answer the model ended early is continued and an exhausted thinking
     budget is closed and answered (config.COMPLETE_ANSWERS); malformed JSON and answers cut off at the token cap are
     resampled; JSON arrays are bounded; empty content fails closed; presence_penalty and reasoning_effort are sent;
  3. execution: the program in every evaluated draft is run on the task instances, and its performance
     (tasks.performance) replaces the LLM feasibility score: Q = performance + NB_TIEBREAK * non-obviousness -
     RUNTIME_TIEBREAK * total solve seconds;
  4. acceptance: a valid program beating the baseline on every instance, a soundness panel that is not fatal (one
     judgment: the program implements the proposed method and both are meant for this task), and the native diversity
     gate; a non-fatal miss is repairable and every accepted idea is refined further for score;
  5. bug fixing (config.CRASH_FIX_*): a program that crashes with a programming bug goes back to the ideator in
     single-turn calls (the current code and its traceback only), separately from repair and refinement;
  6. memory and diversity: empty drafts reach no model and stay out of memory; a candidate is not judged against
     rejected drafts of its own lineage; while nothing is accepted or historically qualified the diversity gate passes;
     a low diversity score that names no compared idea is raised to the gate; critics see the best scores so far;
  7. elite replacement across lineages.
main() then runs scripts/run_ideagent.py exactly as an open-ended IDEAgent run does.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1]
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

RESAMPLE_RETRIES = 2      # malformed JSON or an answer cut off at the token cap: resample those samples only
MAX_ARRAY_ITEMS = 64
EOS_IDS = {248044, 248046}   # <|endoftext|>, <|im_end|> (Qwen3.6 and Qwen3.8 tokenizer)
MAX_CONTINUATIONS = 4        # extra segments per choice (continuations plus at most one forced answer)
FORCED_ANSWER_TOKENS = 6144  # answer room beyond the budget (natural answers take ~2k-4k tokens)
# Qwen's documented thinking-budget closing line (Qwen3 model card), appended only when thinking used the whole budget.
FORCE_ANSWER_TEXT = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
                     "directly now.\n</think>\n\n")


def json_ok(text: str) -> bool:
    """The same first-'{' to last-'}' extraction the native parsers use."""
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end < start:
        return False
    try:
        return isinstance(json.loads(text[start:end + 1]), dict)
    except ValueError:
        return False


def bounded_schema(node):
    """maxItems on arrays. Strings get no maxLength: this vLLM's xgrammar compiles a length-bounded string without
    escapes, so it could not hold a newline, an escaped quote or a backslash."""
    if isinstance(node, list):
        return [bounded_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {k: bounded_schema(v) for k, v in node.items()}
    if out.get("type") == "array" and "maxItems" not in out:
        out["maxItems"] = MAX_ARRAY_ITEMS
    return out


def presence_penalty(model_id: str, temperature) -> float | None:
    """Model-card presence penalty for the sampling mode (native QD forwards only temperature/top_p/top_k)."""
    from verifiable.program_qd import config
    rule = config.PRESENCE_PENALTY.get(model_id)
    return None if rule is None or temperature is None else rule.get(round(float(temperature), 2))


def install_client_guards(event) -> None:
    from ideagent import clients

    from verifiable.program_qd import config, prompts, tasks
    native_request = clients.VLLMOpenAIClient._chat_request

    def chat_request(self, messages, kwargs):
        request = native_request(self, messages, kwargs)
        # A call may set its own effort (the bug-fix calls do); native QD never passes one.
        effort = kwargs.get("reasoning_effort") or config.REASONING_EFFORT.get(self.model_id)
        if effort and kwargs.get("enable_thinking", True) and "reasoning_effort" not in request:
            request["reasoning_effort"] = effort   # native QD does not send it; the chat template reads it
        return request

    clients.VLLMOpenAIClient._chat_request = chat_request

    native_create = clients.VLLMOpenAIClient._chat_completion_create
    force_ids: dict = {}

    def chat_completion_create(self, **request):
        """Every free-text thinking call returns an answer, without changing any response that finishes on its own.
        Qwen3.8 often samples <|im_end|> while its thinking is still open (natively: empty answer, the whole call
        silently retried up to 5 times) and sometimes thinks through the whole budget (natively: empty answer, draft
        lost). Instead:
          - an end-of-turn inside open thinking, an unclosed code block, or (program proposals) no solve() program yet
            is continued from the same tokens (minus that end token), within the request's budget;
          - thinking that reaches the budget keeps all of it; Qwen's documented closing line and </think> are
            appended, and the answer gets FORCED_ANSWER_TOKENS on top of the budget (an answer cut at the budget gets
            the same extra room).
        Structured-output calls keep their grammar and native handling."""
        thinking = ((request.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking", True)
        if not thinking or request.get("response_format") is not None:
            return native_create(self, **request)
        extra = {**(request.get("extra_body") or {}), "return_token_ids": True}
        completion = native_create(self, **{**request, "extra_body": extra})
        prompt_ids = getattr(completion, "prompt_token_ids", None)
        if not prompt_ids:
            return completion
        system = str((request.get("messages") or [{}])[0].get("content", ""))
        wants_program = system.startswith(prompts.IDEATOR_SYSTEM[:40])         # the ideator (critics see the contract too)
        for index, choice in enumerate(completion.choices):
            content = choice.message.content or ""
            if choice.finish_reason == "length" or (choice.finish_reason == "stop" and unfinished(content, wants_program)):
                complete_choice(self, request, extra, prompt_ids, completion, index, choice, wants_program)
        return completion

    def unfinished(content: str, wants_program: bool) -> bool:
        """No answer yet, an unclosed code block, or (program proposal) no solve() program yet."""
        return (not content.strip() or content.count("```") % 2 == 1
                or (wants_program and tasks.extract_program(content) is None))

    def forcing_ids(self, model: str) -> list[int]:
        if model not in force_ids:
            body = json.dumps({"model": model, "prompt": FORCE_ANSWER_TEXT, "add_special_tokens": False}).encode()
            req = urllib.request.Request(f"http://localhost:{self.port}/tokenize", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as response:
                force_ids[model] = json.load(response)["tokens"]
        return force_ids[model]

    def complete_choice(self, request, extra, prompt_ids, completion, index, choice, wants_program) -> None:
        message = choice.message
        reasoning_key = next((k for k in ("reasoning_content", "reasoning") if getattr(message, k, None)), "reasoning_content")
        reasoning, content = getattr(message, reasoning_key, None) or "", message.content or ""
        ids = list(getattr(choice, "token_ids", None) or [])
        budget = int(request.get("max_completion_tokens") or 0)
        limit = budget + FORCED_ANSWER_TOKENS              # thinking stays within budget; the answer may go beyond
        sampling = {k: request[k] for k in ("temperature", "top_p", "presence_penalty") if request.get(k) is not None}
        body = {**{k: v for k, v in extra.items() if k != "chat_template_kwargs"}, "skip_special_tokens": False}
        added, continued, forced = 0, 0, False
        while ids and continued + forced < MAX_CONTINUATIONS:
            thinking_open = not content.strip() and not forced
            if choice.finish_reason == "stop" and ids[-1] in EOS_IDS and unfinished(content, wants_program):
                if thinking_open and len(ids) < budget:
                    prompt, room = list(prompt_ids) + ids[:-1], budget - len(ids) + 1     # resume the thinking
                elif not thinking_open and len(ids) < limit:
                    prompt, room = list(prompt_ids) + ids[:-1], limit - len(ids) + 1      # resume the answer
                elif thinking_open:
                    ids = ids[:-1]
                    prompt, room = None, 0
                else:
                    break
            elif choice.finish_reason == "length":                     # budget reached: force or extend the answer
                prompt = None if thinking_open else list(prompt_ids) + ids
                room = limit - len(ids)
            else:
                break
            if prompt is None:                                                         # budget spent thinking
                prompt, room = list(prompt_ids) + ids + forcing_ids(self, request["model"]), FORCED_ANSWER_TOKENS
                reasoning += FORCE_ANSWER_TEXT.split("</think>")[0]
                content, forced = "", True
            else:
                continued += 1
            if room <= 0:
                break
            more = self.client.completions.create(model=request["model"], prompt=prompt, max_tokens=room,
                                                  **sampling, extra_body=body)
            piece = more.choices[0]
            new_ids = list(getattr(piece, "token_ids", None) or [])
            text = piece.text or ""
            for token in ("<|im_end|>", "<|endoftext|>"):
                text = text[:-len(token)] if text.endswith(token) else text
            if forced and not content:                     # the answer after the appended </think>
                content = text.lstrip("\n")
            elif content:                                  # stopped inside the answer: extend it
                content += text
            elif "</think>" in text:                       # thinking closes in this segment
                head, tail = text.split("</think>", 1)
                reasoning += head
                content = tail.lstrip("\n")
            else:                                          # still thinking
                reasoning += text
            ids = prompt[len(prompt_ids):] + new_ids
            added += len(new_ids)
            choice.finish_reason = piece.finish_reason
        message.content = content or None
        setattr(message, reasoning_key, reasoning)
        if getattr(completion, "usage", None) is not None:
            completion.usage.completion_tokens += added
        event("answer_completed", model=self.model_id, choice=index, continued=continued, forced=forced,
              added_tokens=added, total_tokens=len(ids), finish=choice.finish_reason, content_chars=len(content),
              has_program=tasks.extract_program(content) is not None,
              content_sha256=hashlib.sha256(content.strip().encode()).hexdigest())

    if config.COMPLETE_ANSWERS:
        clients.VLLMOpenAIClient._chat_completion_create = chat_completion_create

    original_many = clients.VLLMOpenAIClient.generate_many

    def guarded(self, messages, n, kwargs):
        """Outputs of one request; (None, text) marks a choice cut off at the token cap."""
        try:
            outputs = list(original_many(self, messages, n=n, **kwargs))
        except ValueError as exc:
            # A choice that ends normally but carries no answer content (all text went to reasoning). Natively this
            # raises after the client's retries and crashes the caller; treat it like an exhausted budget.
            if "did not contain message content" not in str(exc):
                raise
            event("empty_content_fail_closed", model=self.model_id, n=n, reason=str(exc))
            return [""] * int(n)
        reasons = (self.get_last_response_metadata() or {}).get("finish_reasons") or []
        cut = [i for i, reason in enumerate(reasons[:len(outputs)]) if reason == "length"]
        if cut:
            # Natively only an empty cut-off answer raises; a non-empty one passes as if complete.
            event("completion_truncated", model=self.model_id, n=n, truncated=len(cut),
                  max_new_tokens=kwargs.get("max_new_tokens"), tails=[(outputs[i] or "")[-120:] for i in cut[:2]])
            for i in cut:
                outputs[i] = (None, outputs[i])
        return outputs

    def generate_many(self, messages, *, n: int = 1, **kwargs):
        penalty = presence_penalty(self.model_id, kwargs.get("temperature"))
        if penalty is not None and "presence_penalty" not in kwargs:
            kwargs = {**kwargs, "presence_penalty": penalty}
        json_mode = kwargs.get("response_format") is not None
        if json_mode and isinstance(kwargs["response_format"], dict) and "json_schema" in kwargs["response_format"]:
            response_format = json.loads(json.dumps(kwargs["response_format"]))
            response_format["json_schema"]["schema"] = bounded_schema(response_format["json_schema"].get("schema", {}))
            kwargs = {**kwargs, "response_format": response_format}
        outputs = guarded(self, messages, n, kwargs)

        def bad(text):
            return isinstance(text, tuple) or (json_mode and not json_ok(text))

        for attempt in range(1, RESAMPLE_RETRIES + 1):
            redo = [i for i, text in enumerate(outputs) if bad(text)]
            if not redo:
                break
            event("resample", model=self.model_id, attempt=attempt, bad_samples=len(redo), n=n, json_mode=json_mode,
                  truncated=sum(isinstance(outputs[i], tuple) for i in redo))
            for i, text in zip(redo, guarded(self, messages, len(redo), kwargs)):
                outputs[i] = text
        kept = [i for i, text in enumerate(outputs) if isinstance(text, tuple)]
        if kept:
            event("completion_truncated_kept", model=self.model_id, n=n, kept=len(kept))
        return [text[1] if isinstance(text, tuple) else text for text in outputs]

    clients.VLLMOpenAIClient.generate_many = generate_many


NOT_BUGS = {"AssertionError", "MemoryError", "SystemExit", "KeyboardInterrupt"}   # self-checks and limits
FRAME = re.compile(r'^\s*File "([^"]+)", line \d+')


def programming_bug(case: dict) -> bool:
    """The program raised (status 'error') a programming error: an exception that is not the program's own assert or
    raise statement (the program reporting that its method failed) and not a memory error. Timeouts and invalid
    constructions are not bugs."""
    if case.get("valid") or case.get("status") != "error":
        return False
    name = re.match(r"([A-Za-z_]\w*)\b", case.get("reason") or "")
    if not name or name.group(1) in NOT_BUGS or not (name.group(1).endswith(("Error", "Exception"))
                                                       or name.group(1) == "StopIteration"):
        return False
    lines, source = (case.get("error_trace") or "").splitlines(), ""
    for i, line in enumerate(lines):                    # the last frame inside the candidate's own program
        if FRAME.match(line) and FRAME.match(line).group(1).endswith("program.py"):
            nxt = next((l.strip() for l in lines[i + 1:] if l.strip() and set(l.strip()) - set("^~ ")), "")
            source = "" if FRAME.match(nxt) else nxt
    return not source.startswith(("raise ", "raise(", "assert ", "assert("))


def changed_lines(before: str, after: str) -> int:
    """Non-blank lines removed plus added between two versions of a program."""
    diff = difflib.unified_diff([l.rstrip() for l in before.splitlines()], [l.rstrip() for l in after.splitlines()],
                                lineterm="", n=0)
    return sum(1 for l in diff if l[:1] in "+-" and not l.startswith(("+++", "---")) and l[1:].strip())


def bug_report(run: dict) -> dict:
    case = next(c for c in run["cases"] if programming_bug(c))
    instance = ", ".join(f"{k}={v}" for k, v in case["problem"].items() if k != "time_limit_s") or "the instance"
    return {"instance": instance, "error": (case["reason"] or "")[:600],
            "trace": (case.get("error_trace") or case.get("traceback_tail") or "")[-1500:]}


def fix_program(task: str, code: str, run: dict, folder: Path, budget_s: float, ask, *, attempts: int,
                max_changed: int, max_total_changed: int, log=lambda **fields: None) -> tuple[str, dict, dict]:
    """While the run shows a programming bug, up to `attempts` single-turn calls ask(program, bug report), each given only
    the current program and its exception, returning a corrected program or None. A fix that returns no program, returns
    it unchanged, changes more than max_changed non-blank lines in one call or more than max_total_changed lines of the
    original program in total is not used (the call still counts). Each used fix is run in folder/fix<k>.
    Returns the last program that was run, its run, and a summary ('unfixed': a bug is still there)."""
    from verifiable.program_qd import tasks
    info = {"attempts": 0, "errors": [], "changed_lines": 0, "folder": None}
    current = code
    while info["attempts"] < attempts and any(programming_bug(c) for c in run["cases"]):
        info["attempts"] += 1
        k = info["attempts"]
        report = bug_report(run)
        info["errors"].append(report["error"][:300])
        proposed = ask(current, report)
        if proposed is None or proposed.strip() == current.strip():
            log(attempt=k, result="not used: " + ("no program" if proposed is None else "unchanged"),
                error=report["error"][:300])
            continue
        step, total = changed_lines(current, proposed), changed_lines(code, proposed)
        if step > max_changed or total > max_total_changed:
            log(attempt=k, result="not used: too many changes", changed_lines=step, total_changed_lines=total,
                error=report["error"][:300])
            continue
        run = tasks.run_program(task, proposed, folder / f"fix{k}", budget_s)
        current = proposed
        info.update(changed_lines=total, folder=f"fix{k}")
        still = any(programming_bug(c) for c in run["cases"])
        log(attempt=k, result="still crashing" if still else "fixed", changed_lines=step, total_changed_lines=total,
            error=report["error"][:300], all_valid=run["all_valid"], scores=run["scores"])
    info["unfixed"] = any(programming_bug(c) for c in run["cases"])
    info["fixed"] = info["attempts"] > 0 and not info["unfixed"]
    return current, run, info


def run_evidence(task: str, run: dict | None, baseline: list[float]) -> str:
    """The measured run, per instance: validity, score against the baseline and runtime, or the error."""
    from verifiable.program_qd import tasks
    if run is None:
        text = "the proposal contains no ```python block defining solve(problem), so nothing could be run"
        if tasks.suite(task) in tasks.MODULES:
            from verifiable.program_qd import prompts
            text = prompts._edit(text, tasks.MODULES[tasks.suite(task)].PROMPT_EDITS)
        return text
    lines = []
    for case, base in zip(run["cases"], baseline):
        label = ", ".join(f"{k}={v}" for k, v in case["problem"].items() if k != "time_limit_s") or "instance"
        if case["valid"]:
            verdict = "beats" if tasks.better(task, case["score"], base) else "does NOT beat"
            lines.append(f"{label}: valid, score {case['score']:.6g} ({verdict} baseline {base:.6g}), "
                         f"runtime {case['runtime_s']:.1f} s of {case['problem']['time_limit_s']:g} s")
        else:
            error = case["reason"] + (f" | traceback: {case['traceback_tail'][-400:]}" if case["traceback_tail"] else "")
            lines.append(f"{label}: INVALID ({error})")
    return "the proposal's program was run as written: " + "; ".join(lines)


def best_scores(task: str, best: list, run: dict | None) -> list:
    """Per instance, the best valid score of `best` (None: none yet) and `run`."""
    from verifiable.program_qd import tasks
    if run is None:
        return best
    return [case["score"] if case["valid"] and (b is None or tasks.better(task, case["score"], b)) else b
            for case, b in zip(run["cases"], best)]


def best_text(task: str, best: list, baseline: list[float]) -> str:
    """'n=26: 2.63 (baseline 2.5); ...' for the instances with a valid score so far; '' if there is none."""
    from verifiable.program_qd import tasks
    labels = [", ".join(f"{k}={v}" for k, v in p.items() if k != "time_limit_s") or "instance"
              for p in tasks.instances(task, 1.0)]
    return "; ".join(f"{label}: {b:.6g} (baseline {base:.6g})"
                     for label, b, base in zip(labels, best, baseline) if b is not None)


def unattributed(verdict, pool_ids: set[str]) -> bool:
    """A diversity verdict that names no duplicate and no compared (accepted or historical) idea among its closest
    entries: a low score is then not measured against those pools, the only ones diversity counts. Ids are found as
    whole words, so "i004", "[i004]" and "[i004]-[i007]" all name i004."""
    text = " ".join(str(c.get("id", "")) for c in verdict.closest if isinstance(c, dict))
    named = any(re.search(rf"\b{re.escape(pid)}\b", text) for pid in pool_ids)
    return verdict.parsed and not verdict.is_duplicate and not named


def run_note(task: str, run: dict | None, perf: int, baseline: list[float], beats: bool = False) -> str:
    """Native-format feedback note: 'feasibility=<score>; evidence: ...; requested repair: ...'."""
    if run is None:
        repair = "include the complete program as the contract requires"
    elif not run["all_valid"]:
        repair = "fix the failing step so every instance returns a valid construction within time_limit_s"
    elif not beats:
        repair = "the program does not score above the baseline on every instance; improve the step that limits the score"
    else:
        repair = "raise the score further within the same method, keeping every output valid"
    return f"feasibility={perf}; evidence: {run_evidence(task, run, baseline)}; requested repair: {repair}."


def install(log_dir: Path, run_dir: Path, task: str, baseline: list[float], budget_s: float) -> dict:
    from ideagent import agents, generation_loop
    from ideagent.archive import Candidate

    from verifiable.program_qd import config, tasks

    log_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def event(kind: str, **fields) -> None:
        with lock, (log_dir / "overlay_events.jsonl").open("a") as handle:
            handle.write(json.dumps({"at": time.time(), "event": kind, **fields}, default=str) + "\n")

    # 1. Program-task prompts (prompts.py) in place of the native ones: module globals the native code reads at call
    #    time. Every evaluator and critic gets the task background; the directives for repairs and refinements also
    #    carry the parent's measured run.
    from ideagent import agent_prompts, diversity_judge, signature
    from ideagent import prompts as native_prompts

    from verifiable.program_qd import prompts
    if tasks.suite(task) in tasks.MODULES:  # the suite's program contract (ALE-Bench: C++)
        prompts.adapt(tasks.MODULES[tasks.suite(task)].PROMPT_EDITS)
        tasks.use_language([task])
    topics = [json.loads(line) for line in (run_dir / "topics.jsonl").read_text().splitlines() if line.strip()]
    background = next(t for t in topics if t["topic_id"] == task)["background_papers"][0]["full_text"]
    run_notes: dict[str, str] = {}          # sha256 of a draft's text -> its measured run (for revision directives)
    lineage_of: dict[str, str] = {}         # draft id -> lineage id
    best: list = [None] * len(baseline)     # best valid score per instance measured so far in this task (any draft)

    def note_for(idea_text: str) -> str:
        return run_notes.get(hashlib.sha256((idea_text or "").strip().encode()).hexdigest(), "")

    agents.IDEATOR_SYSTEM_PROMPT = prompts.IDEATOR_SYSTEM
    generation_loop.op_free, generation_loop.op_pivot = prompts.op_free, prompts.op_pivot
    generation_loop.op_critic_revision = lambda idea_text, critic_challenge, *, mode: prompts.op_critic_revision(
        idea_text, critic_challenge, mode=mode, run_note=note_for(idea_text))
    native_prompts.NON_OBVIOUSNESS_PRESSURE = prompts.NOVELTY       # read by the native directive builders
    agents.build_feedback_system_prompt = lambda *args, **kwargs: prompts.feedback_system(background)
    agent_prompts._RUBRIC_DIRECTIVE = prompts.RUBRIC_DIRECTIVE
    agent_prompts._GUIDANCE_REQUEST_DIRECTIVE = prompts.GUIDANCE_DIRECTIVE
    agents.SOUNDNESS_EVAL_SYSTEM_PROMPT = prompts.soundness_system(background)
    generation_loop.REPAIR_CRITIC_SYSTEM = prompts.repair_critic_system(background)
    native_critic_user = generation_loop.build_repair_critic_user

    def critic_user(*args, **kwargs):
        """The native repair/refinement critic message plus the best scores measured so far in this task."""
        text, measured = native_critic_user(*args, **kwargs), best_text(task, best, baseline)
        if not measured:
            return text
        block = f"BEST SCORES MEASURED IN THIS TASK SO FAR (any idea; ground truth):\n{measured}\n\n"
        head, sep, tail = text.rpartition("Return exactly one focused repair challenge.")
        return head + block + sep + tail if sep else text + "\n\n" + block.rstrip()

    generation_loop.build_repair_critic_user = critic_user
    generation_loop.NOVELTY_BRANCH_CRITIC_SYSTEM = prompts.novelty_branch_critic_system(background)
    generation_loop.RELATIVE_NB_SYSTEM = prompts.relative_nb_system(background)
    generation_loop.CONSOLIDATION_SYSTEM = prompts.CONSOLIDATION_SYSTEM
    diversity_judge.DIVERSITY_JUDGE_SYSTEM = prompts.diversity_judge_system(background)
    signature.SIGNATURE_SYSTEM = prompts.SIGNATURE_SYSTEM

    # 2. Transport guards.
    install_client_guards(event)

    # 3. Q includes the program's performance (stored in the feasibility slot).
    def runtime_s(candidate) -> float:
        run = next((r for r in candidate.rubric_assessment if r.get("id") == "program_run"), None)
        return sum(t for t in (run or {}).get("runtime_s") or [] if t is not None)

    def weighted_quality(self, weights):
        return (self.feasibility + config.NB_TIEBREAK * self.non_obviousness
                - config.RUNTIME_TIEBREAK * runtime_s(self))

    Candidate.weighted_quality = weighted_quality

    # 4. Acceptance: the program runs validly and beats the baseline on every instance, the soundness panel is not
    #    fatal (config.SOUNDNESS_JUDGMENTS judgment(s), fatal at <= 3: the program does not implement the proposed
    #    method, or the proposal or program is not meant for this task), and the native diversity gate passes.
    #    Non-obviousness, clarity and the soundness mean do not gate acceptance; NB breaks ties in Q. The native
    #    relative-NB veto stays on for refinements: a child that scores higher replaces its parent only if a
    #    side-by-side, order-balanced comparison does not find it more obvious, so a non-obvious idea cannot drift into a
    #    standard recipe. Any non-fatal miss is repairable; every accepted idea is refined further for score.
    def beats(candidate) -> bool:
        run = next((r for r in candidate.rubric_assessment if r.get("id") == "program_run"), None)
        return bool(run and run.get("beats_baseline"))

    unfixed: set[str] = set()       # drafts whose programming bug survived every fix attempt: discarded, never repaired
    Candidate.passes_soundness = lambda self, t: not self.soundness_fatal
    Candidate.passes_quality = lambda self, t: beats(self)
    Candidate.near_thresholds = lambda self, t: (not self.passes_all(t) and not self.soundness_fatal
                                                 and self.id not in unfixed)

    def defect_text(self, t):
        defects = []
        if self.id in unfixed:
            defects.append(f"failed: the code is unusable (a programming bug survived {config.CRASH_FIX_ATTEMPTS} "
                           "bug-fix calls), so the draft is discarded")
        if self.soundness_fatal:
            defects.append(f"soundness panel: the program does not implement the proposed method, or the proposal "
                           f"is not meant for this task: {self.soundness_rationale}".strip())
        if not beats(self):
            run = next((r for r in self.rubric_assessment if r.get("id") == "program_run"), {})
            defects.append(f"the program has no valid run on every instance, so it does not beat the baseline {baseline} "
                           "(see the run evidence)" if not run.get("all_valid")
                           else f"the program does not beat the baseline {baseline} on every instance "
                                f"(performance {self.feasibility})")
        return "; ".join(defects)

    Candidate.defect_text = defect_text

    def press_deficits(self, candidate):
        if not beats(candidate):
            return []
        return [f"program performance {candidate.feasibility}: raise the measured score further beyond the baseline "
                "within the same central move (a higher score replaces the parent unless the revision is judged more "
                "obvious; between equal scores the less obvious central move ranks higher)"]

    generation_loop.GenerationLoop._press_deficits = press_deficits
    import io
    native_write = generation_loop.GenerationLoop._write_candidate

    def write_candidate(cand_log, cand, verdict, outcome, **kwargs):
        """The native row, with gate_results describing the gates actually applied."""
        row = native_write(io.StringIO(), cand, verdict, outcome, **kwargs)
        gates = row["gate_results"]
        gates.update(baseline_pass=beats(cand), soundness_not_fatal=not cand.soundness_fatal,
                     gates_applied=["eval_ok", "baseline_pass", "soundness_not_fatal", "diversity_parsed",
                                    "not_duplicate", "diversity_pass"])
        gates["full_acceptance_gate_pass"] = all(gates.get(name) is True for name in gates["gates_applied"])
        cand_log.write(json.dumps(row, ensure_ascii=False) + "\n")
        cand_log.flush()
        return row

    generation_loop.GenerationLoop._write_candidate = staticmethod(write_candidate)

    # 5. Run the program on every draft. Native _evaluate's signature and rubric calls start at once; the soundness
    #    panel waits for the run (and any bug fix) so it judges the program as kept, and native _evaluate gets that
    #    panel result at its usual call site; the panel does not see the run. An empty draft (the ideator returned
    #    nothing) is not sent to any model: it fails evaluation as natively, with no invented signature.
    native_evaluate = generation_loop.GenerationLoop._evaluate
    native_panel = generation_loop.evaluate_soundness_multi
    native_signature = generation_loop.extract_signature
    from concurrent.futures import ThreadPoolExecutor

    from ideagent.clients import build_client
    from ideagent.experiment_logging import TracedClient, set_trace_context

    def panel_client_for(feedback_client):
        if not isinstance(feedback_client, TracedClient):
            return feedback_client
        raw = build_client(model_id=feedback_client.model_id, vllm_port=getattr(feedback_client, "port", None),
                           api_key_env="VLLM_API_KEY", request_timeout=3600)
        return TracedClient(raw, logger=feedback_client.trace_logger, role=feedback_client.trace_role)

    def ask_fix(loop, topic_id, idea_id, code, report, attempt):
        """One single-turn bug-fix call to the ideator (its own model and sampling, low reasoning effort), given only
        the current program and its exception: the corrected program, or None."""
        client = loop.ideator.client
        set_trace_context(client, topic_id=topic_id, candidate_id=idea_id, stage=f"bug_fix_{attempt}")
        messages = [{"role": "system", "content": prompts.BUGFIX_SYSTEM},
                    {"role": "user", "content": prompts.bugfix_user(code, report)}]
        try:
            text = client.generate(messages, **{**loop.ideator.gen_kwargs, **config.CRASH_FIX_SETTINGS})
        except Exception as exc:      # a failed call is a failed attempt, never a crashed search
            event("bug_fix_call_failed", topic_id=topic_id, idea_id=idea_id, attempt=attempt,
                  error=f"{type(exc).__name__}: {exc}"[:300])
            return None
        return tasks.extract_program(text or "")

    def execute(loop, idea_id, idea_text, topic_id):
        """The draft's run after up to CRASH_FIX_ATTEMPTS bug fixes: (run or None, proposal text, fix summary or None).
        A used fix replaces only the program block of the proposal."""
        code = tasks.extract_program(idea_text)
        if code is None:
            return None, idea_text, None
        folder = run_dir / "programs" / task / idea_id
        run = tasks.run_program(task, code, folder, budget_s)
        if config.CRASH_FIX_ATTEMPTS <= 0 or not any(programming_bug(c) for c in run["cases"]):
            return run, idea_text, None
        calls = iter(range(1, config.CRASH_FIX_ATTEMPTS + 1))
        try:
            fixed, fixed_run, fix = fix_program(
                task, code, run, folder, budget_s,
                lambda current, report: ask_fix(loop, topic_id, idea_id, current, report, next(calls)),
                attempts=config.CRASH_FIX_ATTEMPTS, max_changed=config.CRASH_FIX_MAX_CHANGED_LINES,
                max_total_changed=config.CRASH_FIX_MAX_TOTAL_CHANGED_LINES,
                log=lambda **fields: event("bug_fix", topic_id=topic_id, idea_id=idea_id, **fields))
        except Exception as exc:     # the correction loop must never crash the search: keep the original run
            event("bug_fix_loop_failed", topic_id=topic_id, idea_id=idea_id, error=f"{type(exc).__name__}: {exc}"[:300])
            return run, idea_text, None
        return fixed_run, (tasks.replace_program(idea_text, fixed) if fixed != code else idea_text), fix

    def evaluate_empty(self, idea_text, **kwargs):
        generation_loop.extract_signature = lambda *args, **kw: None
        generation_loop.evaluate_soundness_multi = lambda *args, **kw: []
        self.quality.evaluate = lambda *args, **kw: {"rubric_assessment": []}
        try:
            return native_evaluate(self, idea_text, **kwargs)
        finally:
            generation_loop.extract_signature, generation_loop.evaluate_soundness_multi = native_signature, native_panel
            del self.quality.evaluate

    def evaluate(self, idea_text, **kwargs):
        topic_id, idea_id = kwargs["topic_id"], kwargs["idea_id"]
        lineage_of[idea_id] = kwargs.get("lineage_id") or idea_id
        fix, final_text = None, idea_text
        if not (idea_text or "").strip():
            candidate = evaluate_empty(self, idea_text, **kwargs)
            run = None
            event("empty_draft", topic_id=topic_id, idea_id=idea_id, operator=kwargs.get("operator"))
        else:
            panel_client = panel_client_for(self.quality.client)
            set_trace_context(panel_client, topic_id=topic_id, candidate_id=idea_id, search_step=kwargs.get("gen"),
                              operator=kwargs.get("operator"), stage="soundness_panel")
            cache_key = hashlib.sha256(f"{topic_id}/soundness".encode()).hexdigest()
            with ThreadPoolExecutor(max_workers=2) as pool:
                program = pool.submit(execute, self, idea_id, idea_text, topic_id)

                def panel_with_run():
                    _, text_, _ = program.result()          # the panel judges the program as it will be kept
                    return native_panel(text_, client=panel_client,
                                        gen_kwargs=self.quality.gen_kwargs, n=self.c.soundness_n,
                                        prompt_cache_key=cache_key)

                panel = pool.submit(panel_with_run)
                generation_loop.evaluate_soundness_multi = lambda *args, **kw: panel.result()
                try:
                    candidate = native_evaluate(self, idea_text, **kwargs)
                finally:
                    generation_loop.evaluate_soundness_multi = native_panel
                run, final_text, fix = program.result()
            if final_text != idea_text:          # the corrected program is the draft's program from here on
                candidate.idea_text = final_text
            if fix and fix["unfixed"]:
                unfixed.add(idea_id)
        best[:] = best_scores(task, best, run)
        perf = tasks.performance(task, run["scores"], baseline) if run else 0
        beats = bool(run and run["all_valid"] and all(tasks.better(task, s, b) for s, b in zip(run["scores"], baseline)))
        note = run_note(task, run, perf, baseline, beats)
        run_notes[hashlib.sha256((final_text or "").strip().encode()).hexdigest()] = run_evidence(task, run, baseline)
        candidate.rubric_assessment = list(candidate.rubric_assessment) + [{
            "id": "program_run", "score": perf, "beats_baseline": beats, "llm_feasibility": candidate.feasibility,
            "evidence": note,
            "scores": run["scores"] if run else None, "all_valid": run["all_valid"] if run else False,
            "runtime_s": [c["runtime_s"] for c in run["cases"]] if run else None}]
        event("program_run", topic_id=topic_id, idea_id=idea_id, operator=kwargs.get("operator"),
              found_program=run is not None, all_valid=run["all_valid"] if run else False,
              scores=run["scores"] if run else None, performance=perf, beats_baseline=beats,
              llm_feasibility=candidate.feasibility,
              bug_fix={k: fix[k] for k in ("attempts", "fixed", "unfixed", "changed_lines", "folder")} if fix else None)
        candidate.feasibility = perf
        candidate.feedback_notes = [n for n in candidate.feedback_notes
                                    if not n.lstrip().lower().startswith("feasibility=")] + [note]
        return candidate

    generation_loop.GenerationLoop._evaluate = evaluate

    # 6. Memory hygiene and the diversity rule. Empty drafts never enter memory. The novelty judge does not see
    #    rejected drafts of the candidate's own lineage (a repair resembles its failed parent by design). Its own
    #    instructions measure diversity against accepted and historically qualified ideas only; while both pools are
    #    empty, the diversity gate passes and the judge is not called (its answer would not count).
    from ideagent.diversity_judge import DiversityVerdict
    from ideagent.archive import Archive
    native_record, native_judge = Archive._record_memory, generation_loop.GenerationLoop._judge

    def record_memory(self, cand, note):
        if (cand.idea_text or "").strip():
            native_record(self, cand, note)

    Archive._record_memory = record_memory
    entry_id = re.compile(r"^\[([^\]]+)\]")

    def judge(self, cand, archive, *, topic_id, exclude_accepted_id=None, exclude_historical_lineage_id=None):
        pools_empty = (not archive.accepted_signatures(exclude_id=exclude_accepted_id)
                       and not archive.historical_signatures(exclude_lineage_id=exclude_historical_lineage_id))
        if cand.eval_ok and pools_empty:      # nothing to compare against: no judge call, the diversity gate passes
            return DiversityVerdict(parsed=True, diversity_score=100)
        own = {i for i, lineage in lineage_of.items() if lineage == cand.lineage_id and i != cand.id}
        pending = [e for e in archive.rejected_pending
                   if not (entry_id.match(e) and entry_id.match(e).group(1) in own)]
        pending = pending[-archive.rejected_pending_cap:]
        summary = "\n".join(part for part in (archive.rejected_families_summary.strip(),
                                              "recent: " + " ;; ".join(pending) if pending else "") if part)
        archive.rejected_summary = lambda: summary
        try:
            verdict = native_judge(self, cand, archive, topic_id=topic_id, exclude_accepted_id=exclude_accepted_id,
                                   exclude_historical_lineage_id=exclude_historical_lineage_id)
        finally:
            del archive.rejected_summary
        if verdict.parsed and not verdict.is_duplicate and verdict.diversity_score < self.c.diversity_D:
            pool_ids = ({s.id for s in archive.accepted_signatures(exclude_id=exclude_accepted_id)}
                        | {s.id for s in archive.historical_signatures(exclude_lineage_id=exclude_historical_lineage_id)})
            if unattributed(verdict, pool_ids):
                event("diversity_unattributed", topic_id=topic_id, idea_id=cand.id, operator=cand.operator,
                      judged=verdict.diversity_score, raised_to=self.c.diversity_D, closest=verdict.closest,
                      rejected_family=verdict.rejected_family)
                verdict.diversity_score = self.c.diversity_D
        return verdict

    generation_loop.GenerationLoop._judge = judge

    # 7. Elite replacement across lineages: a draft (seed, repair or refinement) that passes every gate except
    #    diversity, is judged the same move as accepted idea(s) of other lineages and outranks all of them replaces the
    #    weakest, taking its lineage (natively only a lone named duplicate of a non-refinement could).
    from ideagent import archive as native_archive

    native_consider = Archive.consider
    native_refinement = Archive.consider_accepted_refinement

    def same_move(verdict, t) -> bool:
        return verdict.is_duplicate or verdict.diversity_score < t.diversity_D

    def elite_to_replace(archive, cand, verdict, t, weights, own_lineage):
        """The weakest accepted idea of another lineage that cand is judged the same move as, if cand passes every
        gate but diversity and outranks every such idea; else None."""
        if not (cand.eval_ok and verdict.parsed and cand.passes_all(t) and same_move(verdict, t)):
            return None
        named = " ".join(map(str, list(verdict.duplicate_ids) + [c.get("id", "") for c in verdict.closest
                                                                if isinstance(c, dict) and c.get("same_mechanism")]))
        lineages = {archive._historical_lineages.get(h) for h in verdict.historical_duplicate_ids}
        targets = [c for c in archive.accepted.values() if c.lineage_id != own_lineage
                   and (re.search(rf"\b{re.escape(c.id)}\b", named) or c.lineage_id in lineages)]
        if not targets or not all(cand.outranks(x, weights) for x in targets):
            return None
        return min(targets, key=lambda x: (x.weighted_quality(weights), x.feasibility))

    def replace_elite(archive, cand, incumbent, verdict, reason: str) -> str:
        """Native active-duplicate replacement (archive.consider), for any operator and lineage."""
        own = cand.lineage_id
        cand.diversity_score = verdict.diversity_score
        cand.lineage_id = incumbent.lineage_id
        cand.accepted_refinements_used = incumbent.accepted_refinements_used
        cand.novelty_branches_used = incumbent.novelty_branches_used
        archive._record_feasibility_change(cand, incumbent)
        archive._remember_qualified(cand)
        archive._retire_qualified(incumbent, reason)
        del archive.accepted[incumbent.id]
        archive.accepted[cand.id] = cand
        archive._remember_accepted_version(cand, parent=incumbent, reason=reason)
        event("elite_replaced", idea_id=cand.id, operator=cand.operator, from_lineage=own,
              lineage_id=cand.lineage_id, replaced=incumbent.id, performance=cand.feasibility,
              replaced_performance=incumbent.feasibility, reason=reason)
        return native_archive.REPLACED

    def consider(self, cand, verdict, t, weights=native_archive.EARLY_WEIGHTS, *, repair_parent=None):
        incumbent = elite_to_replace(self, cand, verdict, t, weights, cand.lineage_id)
        if incumbent is not None:
            return replace_elite(self, cand, incumbent, verdict, "v7_elite_replaced")
        return native_consider(self, cand, verdict, t, weights, repair_parent=repair_parent)

    def consider_refinement(self, cand, *, parent_id, verdict, t, weights, nb_not_worse=True):
        parent = self.accepted.get(parent_id)
        incumbent = (elite_to_replace(self, cand, verdict, t, weights, parent.lineage_id)
                     if parent is not None and parent.accepted_refinements_used < self.accepted_refinement_limit
                     else None)
        if incumbent is None:
            return native_refinement(self, cand, parent_id=parent_id, verdict=verdict, t=t, weights=weights,
                                     nb_not_worse=nb_not_worse)
        parent.accepted_refinements_used += 1       # the attempt counts against the parent, as natively
        parent.auxiliary_attempts_used = max(parent.auxiliary_attempts_used, cand.auxiliary_attempts_used)
        return replace_elite(self, cand, incumbent, verdict, "v7_elite_replaced_by_refinement")

    Archive.consider = consider
    Archive.consider_accepted_refinement = consider_refinement
    return {
        "condition": config.CONDITION, "native_source_modified": False, "task": task,
        "baseline_scores": baseline, "ideation_budget_s": budget_s,
        "runtime_tiebreak": config.RUNTIME_TIEBREAK, "nb_tiebreak": config.NB_TIEBREAK,
        "changes": ["program-task prompts (prompts.py) replace the native ideator, directive, rubric, soundness, critic, "
                    "relative-NB, novelty-judge, consolidation and steno prompts; evaluators and critics see the background",
                    "critics are told the measured run outranks every claim",
                    f"resample malformed JSON and answers cut off at the token cap (all roles) up to {RESAMPLE_RETRIES} times; "
                    f"fail closed on empty exhausted thinking budgets; maxItems {MAX_ARRAY_ITEMS} on JSON arrays",
                    "the proposal's program is run on every evaluated draft; its performance (0-100) replaces the LLM "
                    "feasibility score and note; Q = performance + NB_TIEBREAK * NB - RUNTIME_TIEBREAK * solve seconds",
                    "acceptance: valid program beating the baseline on every instance + soundness panel not fatal + "
                    "native diversity gate; NB, clarity and the soundness mean do not gate; NB breaks ties in Q; the "
                    "native relative-NB veto on refinements stays on; a non-fatal miss is repairable; accepted ideas are "
                    "refined for score",
                    "empty drafts skip every model call and stay out of memory; the judge does not see the candidate's own "
                    "lineage among rejected drafts; with no accepted or historical idea the diversity gate passes",
                    f"programming bugs: up to {config.CRASH_FIX_ATTEMPTS} single-turn ideator calls (code + exception "
                    f"only, no critic) fix the bugs without changing the algorithm, at most "
                    f"{config.CRASH_FIX_MAX_CHANGED_LINES} changed lines per call and "
                    f"{config.CRASH_FIX_MAX_TOTAL_CHANGED_LINES} in total, proposal text unchanged, separate from repair "
                    "and refinement; a bug that survives every call fails the draft (no repair)",
                    "a diversity score below the gate that names no duplicate and no compared accepted or "
                    "historical idea among its closest entries is raised to the gate (event diversity_unattributed)",
                    "the repair/refinement critic message carries the best valid score per instance measured so "
                    "far in this task (any draft)",
                    "the soundness panel is one judgment (config.SOUNDNESS_JUDGMENTS) checking only alignment "
                    "(the program implements the proposed method) and task fit; it does not see the run",
                    "signatures (steno) describe what the program does, following the code where text and code differ",
                    "elite replacement across lineages (a draft judged the same move as accepted ideas of other "
                    "lineages that passes every other gate and outranks all of them replaces the weakest; event "
                    "elite_replaced)",
                    "refinements rebuild the parent's construction first; each frozen slot is the best version "
                    "among its seed's drafts; native search budget (10 seeds, repair and refinement limits)"],
        "crash_fix": {"attempts": config.CRASH_FIX_ATTEMPTS, "max_changed_lines": config.CRASH_FIX_MAX_CHANGED_LINES,
                      "max_total_changed_lines": config.CRASH_FIX_MAX_TOTAL_CHANGED_LINES,
                      "settings": config.CRASH_FIX_SETTINGS, "system_prompt": prompts.BUGFIX_SYSTEM},
        "effective_prompts": {"ideator_system": agents.IDEATOR_SYSTEM_PROMPT, "free_directive": prompts.op_free([], ""),
                              "diversity_judge_system": diversity_judge.DIVERSITY_JUDGE_SYSTEM,
                              "relative_nb_system": generation_loop.RELATIVE_NB_SYSTEM,
                              "novelty_branch_critic_system": generation_loop.NOVELTY_BRANCH_CRITIC_SYSTEM,
                              "feedback_system": agents.build_feedback_system_prompt([]),
                              "rubric_directive": agent_prompts._RUBRIC_DIRECTIVE,
                              "soundness_system": agents.SOUNDNESS_EVAL_SYSTEM_PROMPT,
                              "repair_critic_system": generation_loop.REPAIR_CRITIC_SYSTEM,
                              "consolidation_system": generation_loop.CONSOLIDATION_SYSTEM,
                              "signature_system": signature.SIGNATURE_SYSTEM},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--baseline", required=True, help="JSON list of baseline scores per instance")
    parser.add_argument("--budget-s", type=float, required=True)
    args = parser.parse_args()
    snapshot = install(Path(args.log_dir), Path(args.run_dir), args.task, json.loads(args.baseline), args.budget_s)
    snapshot["overlay_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (Path(args.log_dir) / "overlay_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    sys.argv = [str(ARCHIVE / "scripts/run_ideagent.py"), "--config", args.config]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
