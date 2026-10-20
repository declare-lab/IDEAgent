"""Entry point for the IDEAgent search.

    python scripts/run_ideagent.py --config configs/ideagent.yaml

Roles: ideator (generates), critic (focuses repairs/refinements), quality (scores
non-obviousness/clarity/feasibility and hosts the soundness panel), steno (extracts compact
signatures), judge (batched diversity judgment + consolidation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ideagent.agents import CriticAgent, IdeatorAgent, QualityAgent
from ideagent.clients import build_client
from ideagent.experiment_logging import ExperimentLogger, TracedClient
from ideagent.response_logging import AsyncResponseLogger
from ideagent.utils import load_config
from ideagent.generation_loop import Config, run_search

ROLES = ("ideator", "critic", "quality", "steno", "judge")


def _sampling(a: dict) -> dict:
    kw: dict = {
        "enable_thinking": a.get("enable_thinking", False),
        "thinking_effort": a.get("thinking_effort"),
        "max_new_tokens": int(a.get("max_new_tokens", 8192)),
    }
    for key in ("temperature", "top_p", "top_k"):
        if a.get(key) is not None:
            kw[key] = float(a[key]) if key != "top_k" else int(a[key])
    return kw


def _opt_int(v):
    return None if v is None else int(v)


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ModuleNotFoundError:
        pass

    parser = argparse.ArgumentParser(description="IDEAgent ideation search.")
    parser.add_argument("--config", type=Path, default=Path("configs/ideagent.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    data, run = cfg.get("data", {}), cfg.get("run", {})
    client_cfg, agents_cfg = cfg.get("client", {}), cfg.get("agents", {})
    api_key_env = client_cfg.get("api_key_env", "GEMINI_API_KEY")
    timeout = float(client_cfg.get("request_timeout", 3600))

    trace_logger = ExperimentLogger(Path(data["output_dir"]) / "experiment_trace.jsonl")
    clients, gen_kwargs, models = {}, {}, {}
    for role in ROLES:
        a = agents_cfg.get(role) or {}
        model_id = a.get("model_id")
        if not model_id:
            raise ValueError(f"agents.{role}.model_id must be set in {args.config}")
        # Per-role override so roles can mix providers (e.g. ideator on OpenAI, the rest on
        # Gemini) -- falls back to the global client.api_key_env when a role doesn't
        # specify its own, so existing single-provider configs are unaffected.
        role_api_key_env = a.get("api_key_env", api_key_env)
        raw_client = build_client(
            model_id=model_id, vllm_port=int(a.get("port", 8000)),
            api_key_env=role_api_key_env, request_timeout=timeout,
        )
        clients[role] = TracedClient(raw_client, logger=trace_logger, role=role)
        gen_kwargs[role] = _sampling(a)
        models[role] = model_id
    print("Agents:")
    for role in ROLES:
        print(f"  {role:<9}: {models[role]}")

    config = Config(
        input_jsonl=data["input_jsonl"], output_dir=data["output_dir"],
        max_background_tokens=_opt_int(run.get("max_background_tokens")),
        resume=bool(run.get("resume", False)),
        init_topic_idx=int(run.get("init_topic_idx", 0)),
        end_topic_idx=_opt_int(run.get("end_topic_idx")),
        max_budget=int(run.get("max_budget", 10)),
        forced_exploration_steps=int(run.get("forced_exploration_steps", 0)),
        soundness_n=int(run.get("soundness_n", 5)),
        soundness_floor=float(run.get("soundness_floor", 60)),
        soundness_fatal_threshold=int(run.get("soundness_fatal_threshold", 3)),
        soundness_fatal_min_votes=_opt_int(run.get("soundness_fatal_min_votes")),
        nb_min=int(run.get("nb_min", 60)), clarity_min=int(run.get("clarity_min", 60)),
        feasibility_actionable_min=int(run.get("feasibility_actionable_min", 60)),
        diversity_D=int(run.get("diversity_D", 55)),
        repair_margin=int(run.get("repair_margin", 20)),
        repair_diversity_D=int(run.get("repair_diversity_D", 45)),
        accepted_capacity=int(run.get("accepted_capacity", 10)),
        accepted_target=_opt_int(run.get("accepted_target")),
        repair_capacity=int(run.get("repair_capacity", 1)),
        repair_attempt_limit=int(run.get("repair_attempt_limit", 1)),
        accepted_refinement_limit=int(run.get("accepted_refinement_limit", 2)),
        auxiliary_attempt_limit=int(run.get("auxiliary_attempt_limit", 2)),
        novelty_branch_limit=int(run.get("novelty_branch_limit", 0)),
        press_nb_target=int(run.get("press_nb_target", 90)),
        press_soundness_target=float(run.get("press_soundness_target", 80)),
        press_clarity_target=int(run.get("press_clarity_target", 80)),
        press_feasibility_target=int(run.get("press_feasibility_target", 0)),
        rejected_pending_cap=int(run.get("rejected_pending_cap", 20)),
        consolidate_every=int(run.get("consolidate_every", 12)),
        signature_char_cap=int(run.get("signature_char_cap", 240)),
        history_field_char_cap=int(run.get("history_field_char_cap", 120)),
        before_full_free=float(run.get("before_full_free", 0.55)),
        before_full_repair=float(run.get("before_full_repair", 0.45)),
        before_full_novelty_branch=float(run.get("before_full_novelty_branch", 0.0)),
        before_full_accepted_refine=float(run.get("before_full_accepted_refine", 0.0)),
        after_full_free=float(run.get("after_full_free", 0.20)),
        after_full_repair=float(run.get("after_full_repair", 0.45)),
        after_full_novelty_branch=float(run.get("after_full_novelty_branch", 0.0)),
        after_full_accepted_refine=float(run.get("after_full_accepted_refine", 0.35)),
        repair_fallback_branch_fraction=float(run.get("repair_fallback_branch_fraction", 0.70)),
        relative_nb_max_retries=int(run.get("relative_nb_max_retries", 2)),
        early_nb_weight=float(run.get("early_nb_weight", 0.70)),
        early_soundness_weight=float(run.get("early_soundness_weight", 0.20)),
        early_clarity_weight=float(run.get("early_clarity_weight", 0.10)),
        late_nb_weight=float(run.get("late_nb_weight", 0.40)),
        late_soundness_weight=float(run.get("late_soundness_weight", 0.40)),
        late_clarity_weight=float(run.get("late_clarity_weight", 0.20)),
        repair_priority_novelty_weight=float(run.get("repair_priority_novelty_weight", 0.50)),
        repair_priority_closeness_weight=float(run.get("repair_priority_closeness_weight", 0.35)),
        repair_priority_remaining_weight=float(run.get("repair_priority_remaining_weight", 0.15)),
        repair_closeness_soundness_weight=float(run.get("repair_closeness_soundness_weight", 0.55)),
        repair_closeness_nb_weight=float(run.get("repair_closeness_nb_weight", 0.15)),
        repair_closeness_clarity_weight=float(run.get("repair_closeness_clarity_weight", 0.30)),
        refinement_priority_novelty_weight=float(run.get("refinement_priority_novelty_weight", 0.60)),
        refinement_priority_headroom_weight=float(run.get("refinement_priority_headroom_weight", 0.25)),
        refinement_priority_remaining_weight=float(run.get("refinement_priority_remaining_weight", 0.15)),
        refinement_headroom_soundness_weight=float(run.get("refinement_headroom_soundness_weight", 0.65)),
        refinement_headroom_clarity_weight=float(run.get("refinement_headroom_clarity_weight", 0.35)),
        seed=int(run.get("seed", 0)),
    )

    response_logger = AsyncResponseLogger(root_path=Path(config.output_dir) / "responses")
    ideator = IdeatorAgent(role="ideator", client=clients["ideator"], gen_kwargs=gen_kwargs["ideator"],
                           response_logger=response_logger, conv_sliding_window=None)
    critic = CriticAgent(role="critic", client=clients["critic"], gen_kwargs=gen_kwargs["critic"],
                         response_logger=response_logger, conv_sliding_window=None)
    quality = QualityAgent(role="quality", client=clients["quality"], gen_kwargs=gen_kwargs["quality"],
                           response_logger=response_logger, conv_sliding_window=None, single_turn=True)

    with response_logger:
        run_search(
            config=config, ideator=ideator, critic=critic, quality=quality,
            steno_client=clients["steno"], steno_gen_kwargs=gen_kwargs["steno"],
            judge_client=clients["judge"], judge_gen_kwargs=gen_kwargs["judge"],
            experiment_logger=trace_logger,
        )


if __name__ == "__main__":
    main()
