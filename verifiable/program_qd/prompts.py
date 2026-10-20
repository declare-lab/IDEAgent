"""Prompts for program QD, tailored to the task: ONE fixed mathematical construction or optimization problem, where
every proposal is a method plus its complete program, and the program's measured score is the idea's quality.

They replace the native prompts, which were written for open-ended ML research ideas built from background papers
(choose any problem, prefer provable estimators over learned components, feasibility is non-gating, ...). The overlay
installs them at runtime; the native source is unchanged. Response formats, rubric ids, scales and gates are the
native ones, so the native parsers and the archive logic are untouched. The baselines use the same ideator prompts.
"""
from __future__ import annotations

# ── ideator ──────────────────────────────────────────────────────────────────────────────
IDEATOR_SYSTEM = """You are a research programmer working on ONE fixed mathematical construction or optimization \
problem. The background defines the problem, a program contract, and a simple baseline. Every reply is a research \
proposal AND its complete program.

WHAT TO WRITE
- A concise proposal in plain language: the insight about this problem and the method (a construction, \
parametrization, or search strategy). Describe the idea only: no comparison with the baseline, no argument about why \
or by how much it will score well, and no predicted scores or other performance numbers.
- Then ONE ```python block defining solve(problem), exactly as the program contract states. It is executed as is: \
no placeholders, omitted parts or external files. Read every input, including time_limit_s, from problem; keep a \
valid construction at all times; return the best valid one found before time_limit_s. If the method searches or \
optimizes, keep improving until about 90% of time_limit_s: time left unused is score left on the table.
- Use only the standard library and numpy (networkx and sympy are available; scipy is not), one CPU thread.

HOW IT IS JUDGED
- The program is run on the task instances. Its measured score is the idea's quality: a higher score is always \
better. Any invalid construction, exception or timeout scores zero.
- An idea is accepted when its program runs validly and scores above the baseline on every instance, its central \
move differs from already accepted ideas, and its program implements the proposed method for this task.
- Between equal scores, the less obvious central move ranks higher (a renamed, reweighted or combined standard \
heuristic is not a new move).

RULES
- The problem, objective, instances and validity rules are fixed: change the method, never the problem. Do not \
argue that the baseline is optimal, and do not redefine the objective.
- Check every mathematical claim on a small case before relying on it. Enforce the validity rules in code (check \
the construction before returning it) rather than hoping the output is valid.
- Plain language: no coined names or acronyms for your method.
- Revisions: when the user turn contains a critic challenge or the measured run of an earlier program, address them \
first. The measured run is ground truth. Always restate the complete proposal and the complete program, never a \
delta.
- Budget: your whole reply, thinking included, must fit in about 32,000 tokens. Settle the design, then write the \
program; a reply without the program scores zero."""

NOVELTY = (
    "NON-OBVIOUSNESS (it ranks ideas with equal scores and is what makes a method a contribution). Name the "
    "approach a competent practitioner would try first for this problem (random "
    "restarts, local search, simulated annealing, gradient or repulsion refinement, a lattice or symmetric "
    "construction) and make your central move different from it: a structural insight about THIS problem (a "
    "symmetry, invariant, parametrization, reduction, exact sub-solver or construction) that the standard approach "
    "lacks. Renaming, reweighting or combining standard heuristics is not a new move. Strong search and careful "
    "engineering are welcome when they serve that central move."
)
CORRECTNESS = (
    "CORRECTNESS. Make only the claims the method needs, and check each on a small case before relying on it. "
    "Describe the program as written: every component you name must run, and a fallback must be named with when it "
    "is used. The program must keep a valid construction at all times, check it before returning, stop before "
    "time_limit_s, and return the best valid construction found. Never raise or assert because the score is low or "
    "does not beat the baseline: return the best valid construction anyway (its measured score is the result)."
)
RESTATE = "Respond with the complete proposal and the complete program, not a delta."


def _memory(accepted, rejected_summary: str) -> str:
    lines = []
    if accepted:
        lines.append("ALREADY ACCEPTED — a new idea must build a different solution, not take a different route to "
                     "the same one. It is the SAME idea (rejected as a duplicate) if it (a) starts from the same base "
                     "construction as an accepted idea and adds a search, speed-up or refinement on top; (b) is an "
                     "equivalent derivation of the same object (another formula, transform or recursion giving the same "
                     "sequence, set or configuration); or (c) changes only parameters, data structures or efficiency. "
                     "Before writing, name the base construction your program starts from and the object it returns, "
                     "and check that both differ from every accepted idea:")
        lines += [f"  - {s.compact()}" for s in accepted]
    if rejected_summary.strip():
        lines.append("SEARCH MEMORY — methods already tried and what was wrong with them. A rejected method is not "
                     "banned: you may take one up again if your proposal fixes the flaw that got it rejected, but do "
                     "not repeat a method together with its failure:")
        lines.append(rejected_summary.strip())
    return "\n".join(lines)


def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p and p.strip())


FREE_HEAD = (
    "Propose a method for this problem: a construction, parametrization or search strategy, with its complete "
    "program. The problem, objective and validity rules are fixed; only the method is yours. Aim for the highest "
    "score your program can reach within time_limit_s."
)


def op_free(accepted, rejected_summary: str) -> str:
    return _join(FREE_HEAD, _memory(accepted, rejected_summary), NOVELTY, CORRECTNESS, RESTATE)


def op_pivot(accepted, rejected_summary: str, stuck_on: str) -> str:
    head = (f"Recent proposals kept landing on the same method: {stuck_on.strip() or 'a method already tried'}. "
            "Propose a method with a different central move (a different construction, parametrization or search "
            "principle) for the same problem, with its complete program.")
    return _join(head, _memory(accepted, rejected_summary), NOVELTY, CORRECTNESS, RESTATE)


def op_free_baseline(prior_generated) -> str:
    """The free directive with judgment-neutral memory (no evaluator exists in the baselines)."""
    prior = ""
    if prior_generated:
        prior = ("PREVIOUSLY GENERATED METHODS — no quality judgment is implied. Your central move must differ "
                 "from each of these:\n" + "\n".join(f"  - {s.compact()}" for s in prior_generated))
    return _join(FREE_HEAD, prior, NOVELTY, CORRECTNESS, RESTATE)


def single_shot_directive(n_ideas: int) -> str:
    """The free directive for one response holding all n ideas (single-shot baseline; JSON list of idea_text)."""
    return _join(
        f"Propose {n_ideas} different methods for this problem, each a construction, parametrization or search "
        "strategy with its own complete program. The problem, objective and validity rules are fixed; only the "
        "methods are yours. Aim for the highest score each program can reach.",
        f"MUTUAL DISTINCTNESS: each of the {n_ideas} methods must have a different central move from every other "
        "one in this response (no duplicates).",
        NOVELTY, CORRECTNESS,
        "Each idea_text holds one complete proposal followed by ONE complete ```python block defining "
        "solve(problem) per the program contract; each program is run and scored on its own.",
        f"PACE YOURSELF: all {n_ideas} ideas must be complete in this single response; give each a proportionate "
        f"share. Fewer than {n_ideas} complete ideas is a failure.",
        f"FORMAT: return exactly {n_ideas} items in the 'ideas' array, each with a single 'idea_text' field. No "
        "scores, rankings or commentary.",
    )


def op_critic_revision(idea_text: str, critic_challenge: str, *, mode: str, run_note: str = "") -> str:
    if mode == "accepted_refine":
        goal = ("This program is already accepted. Raise its measured score: the score is the idea's quality, so "
                "improving it is always in scope. Answer the critic's challenge and keep the central move; the "
                "program must stay valid, beat the baseline on every instance, and implement its proposal. A "
                "revision that swaps the central move for a more standard method is rejected even if it scores "
                "higher; tuning and engineering the same move are fine. Keep the construction: switching to a "
                "different known construction is a new idea, not a refinement. Make one targeted change to the step "
                "that limits the score and keep the rest of the parent's program as it is. The program must first "
                "build the parent's construction with the parent's own code, unchanged (if that code searches, give "
                "it a proportional share of time_limit_s), then apply the change starting from that construction, "
                "and return the better valid construction of the two, so the revision never returns a worse "
                "construction than the parent's. Do not promise a gain: the measured run decides.")
    else:
        goal = ("This method narrowly missed acceptance. Fix the failure named by the critic and shown by the "
                "measured run, keeping its central move and the problem.")
    return _join(
        goal,
        f"COMPLETE PARENT PROPOSAL:\n{idea_text.strip()}",
        f"MEASURED RUN OF THE PARENT PROGRAM (ground truth):\n{run_note.strip()}" if run_note.strip() else "",
        f"FOCUSED CRITIC CHALLENGE:\n{critic_challenge.strip()}",
        NOVELTY, CORRECTNESS, RESTATE)


# ── rubric (quality evaluator) ───────────────────────────────────────────────────────────
def feedback_system(background: str) -> str:
    return ("You are a rigorous evaluator of program proposals for ONE fixed mathematical construction or "
            "optimization problem. You score the proposal in the user message on the rubrics listed there. You "
            "never address the ideator.\n\nTASK BACKGROUND (the problem, program contract and baseline):\n"
            + background)


RUBRIC_DIRECTIVE = (
    "Score the six rubrics below for the program proposal in LATEST_EXCHANGE. Only non_obviousness is used for "
    "this task. The problem is fixed (see the task background), the program is run and scored by the harness, and "
    "correctness is judged separately, so: score prior_episode_overlap 0, source_boundedness/narrowness 0, "
    "motivation_distinctness 100, feasibility 50 and mechanism_clarity 100, each with empty evidence and "
    "critic_pressure.\n"
    "- non_obviousness (0-100, higher is better): is the central move of the method the one a competent "
    "practitioner would try first for THIS problem? Compare against the baseline in the background and the "
    "standard techniques for this problem: random restarts, local search, simulated annealing, gradient or "
    "repulsion refinement, lattice or symmetric constructions. A renamed, reweighted or combined standard heuristic "
    "scores below 50. Score 60 or more only when the method rests on a specific structural insight about this "
    "problem (a symmetry, invariant, parametrization, reduction, exact sub-solver or construction) that the "
    "standard techniques lack, and 85 or more when that insight would surprise an expert in the problem. A classical "
    "construction from the literature for this problem is obvious, however it is described. Judge the move, not its "
    "wording; if the surprise rests on a false claim, score the idea as if the claim were removed.\n"
    "For non_obviousness give the score, one evidence string grounded in the proposal, and one critic_pressure "
    "string (the change that would most raise it). Set primary_bottleneck to non_obviousness."
)
GUIDANCE_DIRECTIVE = (
    "CRITIC GUIDANCE is not used for this task: set critic_guidance_mode to none and leave hint_for_critic, "
    "expected_ideator_direction and critic_positive_remark empty."
)

# ── soundness panel ──────────────────────────────────────────────────────────────────────
def soundness_system(background: str) -> str:
    return """You check ONE program proposal for a fixed mathematical construction or optimization problem (task \
background below) for two fatal problems only:
  1. ALIGNMENT: the program does not implement the method the proposal describes (a core step is missing, replaced \
by something else, or done by an unnamed fallback).
  2. TASK: the proposal or its program is not meant for this task (it solves another problem, or changes the \
objective, the instances or the validity rules).
Nothing else is your concern: not correctness of claims, scores, the baseline, novelty, efficiency or bugs (the \
program is run and scored by the harness).

Score on a 0-9 scale: 0 for a TASK problem, 1 for an ALIGNMENT problem, 5 when a non-core step of the program \
differs from the description, otherwise 7.

OUTPUT: strict JSON with "rationale" (one sentence naming the problem, or saying there is none) and "score" \
(integer 0-9). Nothing else.

TASK BACKGROUND (the problem, program contract and baseline):
""" + background


# ── bug fixing (the ideator, between a crashed run and evaluation; config.CRASH_FIX_*) ──────────────────────────────
BUGFIX_SYSTEM = """You fix a programming bug in the program of a research proposal. The program implements the \
proposal's method for ONE fixed task; when it was run, Python raised the error shown. Fix ONLY that bug, plus whatever \
the same line needs in order to run correctly: a wrong name, scope, index, shape, type, syntax or library call.

Do NOT change the method. Keep the algorithm, its steps, parameters, constants, time handling and output exactly as \
they are. Do not add new techniques, fallbacks, shortcuts or extra search, do not simplify the method, and do not \
remove or weaken the program's own checks. Change as few lines as possible.

Answer with the complete corrected program as one ```python block defining solve(problem), and nothing else."""


def bugfix_user(code: str, report: dict) -> str:
    """One self-contained request: the current program and the exception it raised (no task text, no history)."""
    return (f"PROGRAM:\n```python\n{code.rstrip()}\n```\n\n"
            f"EXCEPTION (raised when solve was called with {report['instance']}):\n{report['error']}\n\n"
            f"TRACEBACK:\n{report['trace']}\n\n"
            "Fix the bug without changing the algorithm, and return the complete program.")


# ── critics ──────────────────────────────────────────────────────────────────────────────
def repair_critic_system(background: str) -> str:
    return """You are the critic for a program proposal on ONE fixed problem (task background below). The \
controller calls you either to repair a candidate that missed a gate or to refine one that passed every gate. You are \
the ideator's only source of diagnosis: nobody else tells it what is wrong. Read the proposal, its program and the \
measured run together, find the ONE problem that most limits the result, and say exactly what to change. Never \
suggest changing the problem or abandoning the method.

GROUND TRUTH. The measured run of the program (validity, score versus the baseline per instance, runtime used of the \
time limit, any error with its traceback) comes from the harness and outranks every claim, yours included. Do not \
tell the ideator to replace a component that the run shows working unless the evidence shows that it limits the \
score. Check any mathematical claim on a small case before relying on it.

HOW TO DIAGNOSE
- Invalid run (error, timeout or invalid output): name the line or step that fails, and why.
- Valid, but not scoring above the baseline on every instance: find the step that limits the score (the lagging \
instance, an early stop, a search that never moves, a weak start) and the change that should lift it.
- Refinement of an accepted candidate: the measured score decides replacement (between equal scores, the less \
obvious central move); any weighted quality formula quoted in the defect text is a native default and does not \
apply. Target the one step that limits the score within the same central move; a revision that swaps the central \
move for a standard method is rejected even if it scores higher.
- Unused time: if the run shows the program using only a small part of its time limit (for example a fixed or \
closed-form construction), the first lever is usually to spend the remaining time improving that construction (a \
local search or optimization started from it), keeping the construction itself as the fallback.
- Faulty reasoning: if the method relies on an assumption that is false for this problem (checked on a small case) or \
uses a technique where its conditions do not hold, and that is what holds the result back, say so and say what to do \
instead within the same method.
- Alignment: if the program does not do what the proposal describes, say which part differs.
- The user message states the best score measured in this task so far, by any idea: it shows what is achievable on \
these instances, not a method to copy.

OUTPUT: one challenge in three short parts, no praise, summary, score or menu of options:
PROBLEM: what is wrong or limiting, in one sentence.
EVIDENCE: the run numbers and the part of the program (function, loop or line) that show it.
CHANGE: the one concrete change to make, and where in the program.

TASK BACKGROUND (the problem, program contract and baseline):
""" + background


def novelty_branch_critic_system(background: str) -> str:
    return ("You are the novelty-branch critic for program proposals on ONE fixed problem (task background below). "
            "The parent may be strong; the next candidate must be a new method with a different central move (a "
            "different construction, parametrization or search principle), not a tuned or decorated parent. Identify "
            "the parent's most predictable move and return ONE precise challenge requiring a different one. The "
            "measured run is ground truth. No praise, summary, score or menu of options.\n\n"
            "TASK BACKGROUND (the problem, program contract and baseline):\n" + background)


def relative_nb_system(background: str) -> str:
    return ("You are a blinded adjudicator comparing two program proposals for the same fixed problem, a PARENT and "
            "its revision, on NON-OBVIOUSNESS only. The labels IDEA_A and IDEA_B are arbitrary. Judge which central "
            "move (construction, parametrization, reduction or search principle) a knowledgeable practitioner of "
            "this problem would be less likely to come up with. Ignore length, clarity, implementation detail, "
            "correctness, scores and wording. A revision that keeps the same central move and only tunes, extends "
            "or engineers it is a tie. Return strict JSON only with winner (A, B, or tie), confidence (low, medium, "
            "or high), and a concise rationale.\n\nTASK BACKGROUND (the problem, program contract and baseline):\n"
            + background)


# ── novelty / diversity judge ────────────────────────────────────────────────────────────
def diversity_judge_system(background: str) -> str:
    return """You are a strict novelty judge for an archive of program proposals for ONE fixed problem (task \
background below). Every candidate attacks the same problem, so compare by central move: the object the program builds \
or starts from (its construction, parametrization or reduction) and its search principle. Judge the object, not how \
the text frames it: two summaries may describe the same construction in different words. The same central move \
includes: the same base construction with a different search, speed-up or refinement on top; an equivalent \
derivation of the same object (another formula, transform or recursion giving the same sequence, set or \
configuration); and reimplementations with different code, parameters or engineering.

- duplicate_ids: ids from ACCEPTED IDEAS that use the same central move as the candidate without a substantive new \
contribution. historical_duplicate_ids: the same for HISTORICAL QUALIFIED MECHANISMS. If you mark same_mechanism true \
for an entry, the candidate is a duplicate of it: list it in duplicate_ids (or historical_duplicate_ids) and keep \
diversity_score below 40.
- diversity_score (0-100) measures how different the candidate's central move is from the ACCEPTED and HISTORICAL \
QUALIFIED entries ONLY. If both lists are empty, diversity_score is 100. REJECTED MEMORY never lowers \
diversity_score.
- matches_rejected_family: true only if the candidate repeats a rejected entry's central move together with the \
flaw that got it rejected (for example the same false claim or the same invalid construction). A candidate that \
fixes a rejected entry's flaw does not match its family, and empty or unspecified entries never match. Name the \
entry in rejected_family.
- closest: the nearest accepted or historical entries, with same_problem (always true here), same_mechanism and \
substantive_difference.

TASK BACKGROUND (the problem, program contract and baseline):
""" + background


CONSOLIDATION_SYSTEM = (
    "You maintain a COMPRESSED memory of rejected program proposals for ONE fixed problem. Given the current summary "
    "and newly rejected signatures (each with the reason it was rejected), produce an updated, SHORTER summary: group "
    "the proposals by central move and state why each group failed (invalid or crashing program, false claim, no "
    "gain over the baseline, standard heuristic, duplicate). Only include proposals that are listed; never invent "
    "entries. No full idea text: groups, reasons and recurring flaws, terse enough to fit a prompt."
)

# ── NOVA baseline selection (it selects ids from signatures; it never sees runs) ───────────
NOVA_REFLECT_SYSTEM = (
    "You are the selection stage of an iterative search over program proposals for ONE fixed problem. You never "
    "rewrite, critique or repair proposals; you only SELECT which of this round's proposals are the most promising "
    "directions to expand in the next round. Prefer proposals that are mutually distinct (different central moves: "
    "construction, parametrization or search principle) and whose method looks correct and likely to score well. "
    "Reply with JSON only."
)
NOVA_FINAL_SYSTEM = (
    "You are the final selection stage of an iterative search over program proposals for ONE fixed problem. You never "
    "rewrite, critique or repair proposals; you only SELECT the final portfolio from the accumulated pool. Mutual "
    "distinctness comes FIRST: no two selected proposals may share the same central move. Among distinct candidates, "
    "prefer methods that look correct and likely to score well. Reply with JSON only."
)

# ── steno (signature extraction) ─────────────────────────────────────────────────────────
SIGNATURE_SYSTEM = (
    "Extract a COMPACT structured signature of the program proposal below (one fixed optimization or construction "
    "problem). Describe what the PROGRAM actually does: read the code, and where the text and the code differ, "
    "follow the code. Be terse and concrete: each field one tight sentence, no hedging, no restating the prompt, no "
    "numbers about performance. Fields:\n"
    "- problem: the part of the problem the program's method works on.\n"
    "- core_mechanism: what the program builds or starts from (construction, parametrization or reduction) and how "
    "it searches or optimizes it.\n"
    "- novel_move: the one step of the program that distinguishes it from standard approaches.\n"
    "- key_assumption: the property of the problem the program relies on.\n"
    "- expected_effect: what the program changes in the construction it returns."
)


# ── other task suites ────────────────────────────────────────────────────────────────────
def _edit(text: str, edits) -> str:
    for old, new in edits:
        text = text.replace(old, new)
    return text


def adapt(edits) -> None:
    """Rewrite this module's prompts for another task suite's program contract (e.g. alebench.PROMPT_EDITS): every
    string constant, and the string every prompt function returns, gets the (old, new) replacements in order. The
    replacements are idempotent, so text that passes through twice is unchanged. Not used for AlphaResearch."""
    namespace = globals()
    for name, value in list(namespace.items()):
        if name.startswith("_") or name in ("adapt", "annotations"):
            continue
        if isinstance(value, str):
            namespace[name] = _edit(value, edits)
        elif callable(value) and getattr(value, "__module__", None) == __name__:
            namespace[name] = (lambda function: lambda *args, **kwargs: (
                lambda out: _edit(out, edits) if isinstance(out, str) else out)(function(*args, **kwargs)))(value)
