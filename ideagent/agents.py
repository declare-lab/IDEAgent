"""No-memory agent wrappers for the multi-turn agentic ideation system.

Each agent owns a growing messages list for the current attempt, optionally capped to the
last conv_sliding_window rounds (FIFO; the system message at index 0 is always kept). Call
reset() / reset_for_topic() at the start of each new attempt to clear history.

Agents:
  IdeatorAgent — generates and refines the research idea
  CriticAgent  — challenges the ideator each round
  QualityAgent — scores non-obviousness/clarity/feasibility and provides private guidance
                 to the critic (soundness is scored separately, see evaluate_soundness_multi)
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ideagent.agent_prompts import (
    CRITIC_SYSTEM_PROMPT,
    IDEATOR_SYSTEM_PROMPT,
    build_critic_kickstart_user_message,
    build_critic_next_user_turn,
    build_feedback_system_prompt,
    build_feedback_user_turn,
    build_ideator_next_user_turn,
    build_ideator_opening_user_message,
)
from ideagent.quality_eval import (
    SOUNDNESS_EVAL_SYSTEM_PROMPT,
    _QUALITY_RESPONSE_FORMAT,
    _build_idea_user_message,
)
from ideagent.request_checkpoint import current as checkpoint_journal


FEEDBACK_RUBRIC_IDS = (
    "prior_episode_overlap",
    "source_boundedness/narrowness",
    "motivation_distinctness",
    "non_obviousness",
    "feasibility",
    "mechanism_clarity",
)
# soundness is no longer part of feedback's own rubric block -- it's scored by a standalone,
# multi-sample dedicated judge (evaluate_soundness_multi) reusing the eval's own rubric,
# since feedback's live self-scored soundness was shown to be unreliable (drifted upward
# round over round regardless of actual mechanism validity).
LOWER_IS_BETTER_RUBRIC_IDS = {"source_boundedness/narrowness", "prior_episode_overlap"}
# In explore phase only these rubrics drive bottleneck selection; quality rubrics are
# deferred to refine+ so they don't create refinement pressure during exploration.
_EXPLORE_PHASE_BOTTLENECK_IDS: frozenset[str] = frozenset({
    "source_boundedness/narrowness",
    "motivation_distinctness",
    "non_obviousness",
    "prior_episode_overlap",
})

_STRING_FIELD_SCHEMA: dict[str, Any] = {"type": "string"}


def _json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
        },
    }


_FEEDBACK_RUBRIC_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id":              {"type": "string", "enum": list(FEEDBACK_RUBRIC_IDS)},
        "evidence":        _STRING_FIELD_SCHEMA,
        "critic_pressure": _STRING_FIELD_SCHEMA,
        "score":           {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["id", "evidence", "critic_pressure", "score"],
    "additionalProperties": False,
}


def _feedback_schema(*, min_rubrics: int, max_rubrics: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rubric_assessment": {
                "type": "array",
                "items": _FEEDBACK_RUBRIC_ITEM_SCHEMA,
                "minItems": min_rubrics,
                "maxItems": max_rubrics,
            },
            "primary_bottleneck": {
                "type": "string",
                "enum": ["", *FEEDBACK_RUBRIC_IDS],
            },
            "hint_for_critic": _STRING_FIELD_SCHEMA,
            "expected_ideator_direction": _STRING_FIELD_SCHEMA,
            "critic_positive_remark": _STRING_FIELD_SCHEMA,
            "critic_guidance_mode": {
                "type": "string",
                "enum": ["blend", "continue", "none"],
            },
            "discussion_phase": {
                "type": "string",
                "enum": ["explore", "refine"],
            },
            "phase_signal_for_critic": _STRING_FIELD_SCHEMA,
        },
        "required": [
            "rubric_assessment",
            "primary_bottleneck",
            "hint_for_critic",
            "expected_ideator_direction",
            "critic_positive_remark",
            "critic_guidance_mode",
            "discussion_phase",
            "phase_signal_for_critic",
        ],
        "additionalProperties": False,
    }


_FEEDBACK_RESPONSE_FORMAT = _json_schema_response_format(
    "agentic_feedback",
    _feedback_schema(min_rubrics=len(FEEDBACK_RUBRIC_IDS), max_rubrics=len(FEEDBACK_RUBRIC_IDS)),
)

_DEFAULT_PHASE_SIGNALS = {
    "explore": (
        "The discussion is still searching. Challenge broadly for novelty, missing dimensions, "
        "and major assumptions."
    ),
    "refine": (
        "The discussion has a plausible direction. Push for mechanism, specificity, feasibility, "
        "and sharper contrast with the background papers."
    ),
}


class _BaseAgent:
    def __init__(
        self,
        *,
        role: str,
        client: Any,
        gen_kwargs: dict[str, Any],
        response_logger: Any | None = None,
        conv_sliding_window: int | None = None,
        single_turn: bool = False,
    ) -> None:
        self.role = role
        self.client = client
        self.gen_kwargs = gen_kwargs
        self.response_logger = response_logger
        self.conv_sliding_window = conv_sliding_window
        self.single_turn = single_turn
        self.messages: list[dict[str, str]] = []
        self.log_context: dict[str, Any] = {}

    def reset(self) -> None:
        self.messages = []

    def set_log_context(self, **metadata: Any) -> None:
        self.log_context = {k: v for k, v in metadata.items() if v is not None}

    def _prompt_cache_key(self) -> str:
        """Deterministic key for OpenAI's prompt_cache_key -- (topic_id, role) ONLY, not any
        per-candidate id. Every generation in the QD/Yield loops is an independent .open()
        call (no .respond(), no growing conversation), but the background/avoidance context
        resent each time is stable for the whole topic -- so all candidates for one topic
        must share ONE cache key to route to the same warm cache. Including a per-candidate
        id (e.g. generation_loop's idea_id, which changes every single call) would give every
        generation its own key and defeat caching entirely, even though the content
        actually repeats. Backends that don't read this kwarg (Gemini, vLLM) ignore it."""
        source = f"{self.log_context.get('topic_id', '')}/{self.role}"
        return hashlib.sha256(source.encode()).hexdigest()

    def _generate(
        self,
        *,
        event: str,
        metadata: dict[str, Any] | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> str:
        output = self.client.generate(
            self.messages,
            **self.gen_kwargs,
            prompt_cache_key=self._prompt_cache_key(),
            **(request_kwargs or {}),
        ).strip()
        if self.response_logger is not None:
            self.response_logger.log(
                role=self.role,
                event=event,
                text=output,
                metadata={**self.log_context, **(metadata or {})},
            )
        return output

    def _append_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self._apply_sliding_window()

    def _append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def _apply_sliding_window(self) -> None:
        """Keep only the last conv_sliding_window (user, assistant) round-pairs, always
        preserving the system message at index 0. FIFO: earlier rounds fall off first."""
        if self.conv_sliding_window is None:
            return
        system_msg, rounds = self.messages[0], self.messages[1:]
        keep = self.conv_sliding_window * 2
        if len(rounds) > keep:
            rounds = rounds[-keep:]
        self.messages = [system_msg, *rounds]


class IdeatorAgent(_BaseAgent):
    def open(
        self,
        *,
        background: str,
        prior_cores_block: str | None,
        critic_opening_challenge: str,
    ) -> str:
        """Round 1: build initial messages and generate first idea."""
        self.messages = [
            {"role": "system", "content": IDEATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_ideator_opening_user_message(
                    background, prior_cores_block, critic_opening_challenge,
                ),
            },
        ]
        idea = self._generate(event="open")
        self._append_assistant(idea)
        return idea

    def respond(self, *, critic_challenge: str, soundness_audit: str | None = None) -> str:
        """Rounds 2+: append critic challenge (+ optional raw soundness audit), return revised idea."""
        self._append_user(build_ideator_next_user_turn(critic_challenge, soundness_audit=soundness_audit))
        idea = self._generate(event="respond")
        self._append_assistant(idea)
        return idea


class CriticAgent(_BaseAgent):
    def focused_review(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return one focused critique for a controller-selected promising candidate.

        Unlike ``open``/``challenge`` in the legacy discussion loop, this is deliberately a
        single critic call: the Yield controller has already evaluated the idea and only invokes
        the critic for a near-threshold repair. The ideator itself retains its conversation and
        receives this critique through ``IdeatorAgent.respond``.
        """
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        critique = self._generate(event="focused_review")
        self._append_assistant(critique)
        return critique

    def open(
        self,
        *,
        background: str,
        prior_cores_block: str | None,
    ) -> str:
        """Round 1: build initial messages and generate opening challenge."""
        self.messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_critic_kickstart_user_message(
                    background, prior_cores_block,
                ),
            },
        ]
        challenge = self._generate(event="open")
        self._append_assistant(challenge)
        return challenge

    def challenge(
        self,
        *,
        ideator_idea: str,
        private_feedback_text: str,
        discussion_phase: str,
        phase_signal: str,
        critic_guidance_mode: str = "",
        soundness_fatal: bool = False,
    ) -> str:
        """Rounds 2+: append ideator idea + feedback, return new challenge."""
        self._append_user(
            build_critic_next_user_turn(
                ideator_idea, private_feedback_text, discussion_phase, phase_signal,
                critic_guidance_mode, soundness_fatal=soundness_fatal,
            )
        )
        new_challenge = self._generate(
            event="challenge",
            metadata={
                "discussion_phase": discussion_phase,
                "has_private_feedback": bool(private_feedback_text.strip()),
            },
        )
        self._append_assistant(new_challenge)
        return new_challenge


class QualityAgent(_BaseAgent):
    def reset_for_topic(
        self,
        *,
        prior_final_idea_cores: list[dict[str, Any]] | None = None,
        summary_only: bool = False,
    ) -> None:
        """Reset and set the system prompt for a new topic."""
        self._has_prior_context = bool(prior_final_idea_cores)
        self.messages = [
            {
                "role": "system",
                "content": build_feedback_system_prompt(
                    prior_final_idea_cores, summary_only=summary_only,
                ),
            },
        ]

    def reset(self) -> None:
        self.messages = []

    def evaluate(
        self,
        *,
        prev_critic_turn: str,
        latest_ideator_turn: str,
        current_final_idea_core: dict[str, Any] | None = None,
        convergence_threshold: int = 75,
        diversity_floor: int = 60,
        non_obviousness_floor: int = 60,
    ) -> dict[str, Any]:
        """Append the latest exchange, generate verdict, append assistant response.
        Bounded by conv_sliding_window like critic/ideator (null = full history kept) --
        unless single_turn is set, in which case each round rebuilds [system, user] fresh,
        with no memory of prior rounds' own scores (removes self-anchoring on its own past
        judgments)."""
        if self.single_turn:
            system_message = self.messages[0]
            user_message = {
                "role": "user",
                "content": build_feedback_user_turn(prev_critic_turn, latest_ideator_turn, current_final_idea_core),
            }
            self.messages = [system_message, user_message]
        else:
            self._append_user(build_feedback_user_turn(prev_critic_turn, latest_ideator_turn, current_final_idea_core))
        raw = self._generate(
            event="evaluate",
            request_kwargs={"response_format": _FEEDBACK_RESPONSE_FORMAT},
        )
        verdict = _parse_feedback(
            raw,
            has_prior_context=self._has_prior_context,
            convergence_threshold=convergence_threshold,
            diversity_floor=diversity_floor,
            non_obviousness_floor=non_obviousness_floor,
        )
        if not self.single_turn:
            self._append_assistant(raw)
        return verdict


def _generation_floor_failures(
    rubrics: list[dict[str, Any]],
    *,
    diversity_floor: int = 60,
    non_obviousness_floor: int = 60,
) -> list[str]:
    """Per-round floor check — independent of tier/phase, no priority between the two.
    Returns the name(s) of whichever gate(s) are below their floor THIS round (empty if all
    clear). Soundness is NOT checked here -- it's scored by a standalone dedicated judge
    (evaluate_soundness_multi), not by feedback's own rubric_assessment. This only detects;
    the result is included in QualityAgent.evaluate()'s return dict but not currently read
    by generation_loop.py. Missing rubric data defaults to "satisfied" so a parsing gap never
    fabricates a failure."""
    by_id = {r["id"]: r for r in rubrics if isinstance(r, dict) and "id" in r}

    def satisfaction(rubric_id: str) -> int:
        r = by_id.get(rubric_id)
        if not r:
            return 100
        score = max(0, min(100, int(r.get("score", 0))))
        return 100 - score if rubric_id in LOWER_IS_BETTER_RUBRIC_IDS else score

    # Diversity is fatal ONLY when the idea is genuinely too similar to prior episodes or too
    # source-bound (a near-duplicate). motivation_distinctness is deliberately EXCLUDED from the
    # hard floor: pressing distinctness as a per-round pivot gate makes the ideator keep jumping
    # to a "more distinct" mechanism every round instead of settling on one and repairing its
    # soundness (observed thrash). Distinctness/novelty is still pushed via the critic's
    # explore-phase prompting -- it just no longer triggers a pivot on its own.
    diversity = min(
        satisfaction("prior_episode_overlap"),
        satisfaction("source_boundedness/narrowness"),
    )
    non_obviousness = satisfaction("non_obviousness")

    failures = []
    if diversity < diversity_floor:
        failures.append("diversity")
    if non_obviousness < non_obviousness_floor:
        failures.append("non_obviousness")
    return failures


def _has_converged(rubrics: list[dict[str, Any]], *, threshold: int = 75) -> bool:
    """True only when all seven rubrics are present THIS round and each clears
    threshold (satisfaction-normalized, same convention as the floor check). Unlike the
    floor check, a missing rubric does NOT default to satisfied here -- an early exit is a
    much more consequential, permanent decision than a per-round pivot warning, so a
    parsing gap must never be able to trigger one."""
    by_id = {r["id"]: r for r in rubrics if isinstance(r, dict) and "id" in r}
    if len(by_id) < len(FEEDBACK_RUBRIC_IDS):
        return False
    for rubric_id in FEEDBACK_RUBRIC_IDS:
        score = max(0, min(100, int(by_id[rubric_id].get("score", 0))))
        satisfaction = 100 - score if rubric_id in LOWER_IS_BETTER_RUBRIC_IDS else score
        if satisfaction < threshold:
            return False
    return True


def _parse_feedback(
    raw: str,
    *,
    has_prior_context: bool,
    convergence_threshold: int = 75,
    diversity_floor: int = 60,
    non_obviousness_floor: int = 60,
) -> dict[str, Any]:
    obj = _extract_json_object(raw)
    if obj is None:
        return {
            "rubric_assessment": [],
            "primary_bottleneck": "",
            "hint_for_critic": "",
            "expected_ideator_direction": "",
            "critic_positive_remark": "",
            "critic_guidance_mode": "none",
            "discussion_phase": "explore",
            "phase_signal_for_critic": _DEFAULT_PHASE_SIGNALS["explore"],
            "floor_failures": [],
            "converged": False,
        }
    rubric_assessment = _parse_rubric_assessment(obj.get("rubric_assessment"))
    if not has_prior_context:
        # No prior episodes or discarded attempts exist yet for this topic -- there is
        # nothing for the model to compare against, so prior_episode_overlap is forced to
        # 0 deterministically rather than trusting the model to self-apply this fallback
        # (it doesn't reliably do so). Computed before primary_bottleneck/floor_failures
        # below so both use the corrected score.
        rubric_assessment = [
            {**r, "score": 0, "evidence": "", "critic_pressure": ""}
            if r.get("id") == "prior_episode_overlap" else r
            for r in rubric_assessment
        ]
    phase = str(obj.get("discussion_phase", "")).strip()
    if phase not in _DEFAULT_PHASE_SIGNALS:
        phase = "explore"
    phase_signal = str(obj.get("phase_signal_for_critic", "")).strip()
    if not phase_signal:
        phase_signal = _DEFAULT_PHASE_SIGNALS[phase]
    primary_bottleneck = _parse_primary_bottleneck(obj.get("primary_bottleneck"), rubric_assessment, phase=phase)
    hint = str(obj.get("hint_for_critic", "")).strip()
    expected_direction = str(obj.get("expected_ideator_direction", "")).strip()
    positive_remark = str(obj.get("critic_positive_remark", "")).strip()
    guidance_mode = str(obj.get("critic_guidance_mode", "")).strip().lower()
    if guidance_mode not in {"blend", "continue", "none"}:
        if hint:
            guidance_mode = "blend"
        elif positive_remark or expected_direction:
            guidance_mode = "continue"
        else:
            guidance_mode = "none"
    if guidance_mode == "blend" and not hint:
        guidance_mode = "continue" if (positive_remark or expected_direction) else "none"
    if guidance_mode == "continue":
        hint = ""
    elif guidance_mode == "none":
        hint = ""
        expected_direction = ""
        positive_remark = ""
    elif guidance_mode == "blend":
        positive_remark = ""

    # Which floor gate(s), if any, are failing THIS round.
    floor_failures = _generation_floor_failures(
        rubric_assessment,
        diversity_floor=diversity_floor,
        non_obviousness_floor=non_obviousness_floor,
    )
    # Whether every rubric clears the convergence bar THIS round.
    converged = _has_converged(rubric_assessment, threshold=convergence_threshold)

    return {
        "rubric_assessment": rubric_assessment,
        "primary_bottleneck": primary_bottleneck,
        "hint_for_critic": hint,
        "expected_ideator_direction": expected_direction,
        "critic_positive_remark": positive_remark,
        "critic_guidance_mode": guidance_mode,
        "discussion_phase": phase,
        "phase_signal_for_critic": phase_signal,
        "floor_failures": floor_failures,
        "converged": converged,
    }


def _parse_score(value: Any) -> int:
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def _parse_rubric_assessment(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        rubric_id = str(item.get("id", "")).strip()
        if rubric_id not in FEEDBACK_RUBRIC_IDS or rubric_id in by_id:
            continue
        by_id[rubric_id] = {
            "id": rubric_id,
            "score": _parse_score(item.get("score", 0)),
            "evidence": str(item.get("evidence", "")).strip(),
            "critic_pressure": str(item.get("critic_pressure", "")).strip(),
        }
    return [by_id[rubric_id] for rubric_id in FEEDBACK_RUBRIC_IDS if rubric_id in by_id]


def _rubric_satisfaction_score(rubric_id: str, score: int) -> int:
    score = max(0, min(100, int(score)))
    if rubric_id in LOWER_IS_BETTER_RUBRIC_IDS:
        return 100 - score
    return score


_TIER1_OVERLAP_THRESHOLD = 25   # prior_episode_overlap score at which Tier 1 wins unconditionally
_TIER2_DISSATISFACTION_MIN = 25  # min dissatisfaction for a Tier 2 rubric to beat Tier 3 in refine+


def _parse_primary_bottleneck(
    raw: Any, rubrics: list[dict[str, Any]], *, phase: str = "explore"
) -> str:
    by_id = {r["id"]: r for r in rubrics if isinstance(r, dict) and "id" in r}

    def dissatisfaction(rubric_id: str) -> int:
        r = by_id.get(rubric_id)
        if not r:
            return 0
        return 100 - _rubric_satisfaction_score(rubric_id, int(r.get("score", 0)))

    # Tier 1: prior_episode_overlap wins whenever it is non-trivial.
    if dissatisfaction("prior_episode_overlap") >= _TIER1_OVERLAP_THRESHOLD:
        return "prior_episode_overlap"

    # Tier 2: exploration diversity rubrics.
    tier2_ids = ("source_boundedness/narrowness", "motivation_distinctness", "non_obviousness")
    tier2_present = [rid for rid in tier2_ids if rid in by_id]
    if tier2_present:
        worst_t2 = max(tier2_present, key=dissatisfaction)
        if phase == "explore":
            return worst_t2  # Tier 2 always beats Tier 3 in explore phase
        if dissatisfaction(worst_t2) >= _TIER2_DISSATISFACTION_MIN:
            return worst_t2

    # Tier 3: quality rubrics — refine+ only.
    if phase != "explore":
        tier3_ids = ("feasibility", "mechanism_clarity")
        tier3_present = [rid for rid in tier3_ids if rid in by_id]
        if tier3_present:
            return max(tier3_present, key=dissatisfaction)

    # Fallback: accept model's own valid choice, else empty.
    candidate = str(raw or "").strip()
    active_ids = (
        _EXPLORE_PHASE_BOTTLENECK_IDS if phase == "explore" else set(FEEDBACK_RUBRIC_IDS)
    )
    return candidate if candidate in active_ids else ""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    candidate = match.group(0) if match else stripped
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def evaluate_soundness_multi(
    ideator_turn: str,
    *,
    client: Any,
    gen_kwargs: dict[str, Any],
    n: int,
    prompt_cache_key: str | None = None,
) -> list[dict[str, Any]]:
    """Standalone, multi-sample soundness check reusing the eval's own dedicated rubric
    (SOUNDNESS_EVAL_SYSTEM_PROMPT) directly, instead of feedback's shared 7-rubric block --
    feedback's live self-scored soundness was shown to drift upward round after round
    regardless of actual mechanism validity, while this same isolated prompt reliably
    caught fatal flaws in post-hoc eval. Runs n independent 0-9-scale judgments in parallel
    on the ideator's latest full turn (already a complete write-up per its own standing
    instructions, so no reconstruction needed). Returns only whatever subset parsed
    successfully -- never fabricates a score for a failed call.

    prompt_cache_key should be constant for the whole topic (e.g. f"{topic_id}/soundness"),
    NOT per-candidate -- every call's user message differs (a different idea's text each
    time), but the system message (SOUNDNESS_EVAL_SYSTEM_PROMPT) is identical across every
    candidate in the topic, so a topic-wide key maximizes how often that shared prefix
    actually gets cache-matched, and the n parallel samples for one candidate (identical
    content) benefit regardless."""
    messages = [
        {"role": "system", "content": SOUNDNESS_EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": _build_idea_user_message({"text": ideator_turn}, "soundness")},
    ]
    call_kwargs = dict(gen_kwargs)
    if prompt_cache_key:
        call_kwargs["prompt_cache_key"] = prompt_cache_key

    def _one_call(index) -> dict[str, Any] | None:
        # With a request journal, each parallel vote is labelled so resume gives it back its own answer.
        slot = {"_checkpoint_slot": ["soundness", index]} if checkpoint_journal() is not None else {}
        raw = client.generate(messages, response_format=_QUALITY_RESPONSE_FORMAT, **slot, **call_kwargs)
        obj = _extract_json_object(raw)
        if obj is None or "score" not in obj:
            return None
        score = max(0, min(9, int(obj.get("score", 0))))
        rationale = str(obj.get("rationale", "")).strip()
        return {"score": score, "rationale": rationale}

    with ThreadPoolExecutor(max_workers=max(1, n)) as executor:
        futures = [executor.submit(_one_call, index) for index in range(n)]
        results = [f.result() for f in futures]
    return [r for r in results if r is not None]
