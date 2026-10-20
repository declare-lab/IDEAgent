"""Freeze every final proposal before any implementation or test feedback."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from .common import ROOT, check_hash, digest, file_digest, json_bytes, load_json, now, write_json
from .registry import known_task


def freeze(sources: dict[str, list[str | Path]], destination: str | Path, *, task_map: dict[str, str] | None = None) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a fresh freeze directory")
    if not destination.is_relative_to(ROOT):
        raise ValueError("Freeze directories must be inside verifiable")
    task_map = task_map or {}
    proposals, source_records, source_blobs, seen_topics = [], [], [], set()
    # Validate all source records before creating an artifact.
    for framework, paths in sources.items():
        if not paths:
            raise ValueError(f"No final files supplied for {framework}")
        for path in paths:
            path = Path(path).resolve()
            source_bytes = path.read_bytes()
            source_hash = digest(source_bytes)
            records = [json.loads(line) for line in source_bytes.decode().splitlines() if line.strip()]
            if not records:
                raise ValueError(f"Empty final file: {path}")
            source_records.append({"framework": framework, "path": str(path), "sha256": source_hash})
            source_blobs.append(source_bytes)
            for record in records:
                topic = record.get("topic_id")
                task = task_map.get(topic, topic)
                if not isinstance(task, str) or not known_task(task):
                    raise ValueError(f"Map topic {topic!r} to a registered task before freezing")
                if (framework, topic) in seen_topics:
                    raise ValueError(f"Duplicate topic for {framework}: {topic}")
                seen_topics.add((framework, topic))
                items = record.get("items")
                if not isinstance(items, list):
                    raise ValueError("Expected final topic items[] with complete idea_text fields")
                expected = record.get("free_seed_budget", record.get("n_requested", len(items)))
                if type(expected) is not int or expected < len(items) or expected < 1:
                    raise ValueError("Invalid declared final proposal budget")
                for index in range(expected):
                    item = items[index] if index < len(items) else {"idea_id": f"missing-{index + 1}", "idea_text": ""}
                    if not isinstance(item, dict) or not isinstance(item.get("idea_text"), str):
                        raise ValueError("Every final item must preserve its full idea_text; compact cores cannot substitute")
                    text = item["idea_text"]
                    identifier = digest(json_bytes([framework, topic, index, item]))[:24]
                    proposals.append({"proposal_id": identifier, "framework": framework, "topic_id": topic, "task": task,
                                      "index": index, "idea_text": text, "idea_sha256": digest(text.encode()),
                                      "status": "ready" if text.strip() else "generation_failed",
                                      "original_item": item,
                                      "origin": {"path": str(path), "sha256": source_hash},
                                      "generation_accounting": {key: value for key, value in record.items() if key != "items"}})
    if not proposals:
        raise ValueError("No proposals")
    destination.mkdir(parents=True)
    for index, source in enumerate(source_records):
        target = destination / "sources" / f"{index:03d}.jsonl"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(source_blobs[index])
        check_hash(target, source["sha256"])
        source["frozen_copy"] = str(target)
    accounting_artifacts = []
    seen_artifacts = set()
    for source in source_records:
        directory = Path(source["path"]).parent
        paths = [directory / "experiment_trace.jsonl", *directory.glob("run_manifest_*.json"), *directory.glob("trace_summary_*.json")]
        for path in sorted(paths):
            if not path.is_file() or path in seen_artifacts:
                continue
            seen_artifacts.add(path)
            content = path.read_bytes()
            target = destination / "accounting" / f"{len(accounting_artifacts):03d}-{path.name}"
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(content)
            accounting_artifacts.append({"framework": source["framework"], "path": str(path),
                                         "sha256": digest(content), "frozen_copy": str(target)})
    bundle = {"schema_version": 1, "frozen_at": now(), "proposal_count": len(proposals),
              "sources": source_records, "accounting_artifacts": accounting_artifacts, "proposals": proposals,
              "policy": "Every native final representative is retained, including rejected lineages and failed/missing output slots. No test-based filtering or extra version selection."}
    path = write_json(destination / "proposals.json", bundle)
    write_json(destination / "freeze.json", {"path": str(path), "sha256": file_digest(path)})
    return {"bundle": str(path), "proposal_count": len(proposals), "sha256": file_digest(path)}


def load_frozen(bundle: str | Path) -> dict:
    bundle = Path(bundle).resolve()
    receipt = load_json(bundle.parent / "freeze.json")
    if Path(receipt["path"]).resolve() != bundle:
        raise ValueError("Freeze receipt path mismatch")
    check_hash(bundle, receipt["sha256"])
    result = load_json(bundle)
    for source in result["sources"]:
        check_hash(source["frozen_copy"], source["sha256"])
    for artifact in result.get("accounting_artifacts", []):
        check_hash(artifact["frozen_copy"], artifact["sha256"])
    for proposal in result["proposals"]:
        if digest(proposal["idea_text"].encode()) != proposal["idea_sha256"]:
            raise ValueError("Proposal text hash mismatch")
    return result


def get_proposal(bundle: str | Path, proposal_id: str) -> dict:
    matches = [p for p in load_frozen(bundle)["proposals"] if p["proposal_id"] == proposal_id]
    if len(matches) != 1:
        raise ValueError("Unknown or nonunique proposal_id")
    return matches[0]


def freeze_run(run_dir: str | Path, destination: str | Path) -> dict:
    run_dir = Path(run_dir).resolve()
    run = load_json(run_dir / "run.json")
    sources = {}
    for framework in run["configs"]:
        from .ideation import generation_command
        generation_command(run_dir, framework)
        paths = sorted((run_dir / "generations" / framework).glob("final_idea_cores*.jsonl"))
        observed = []
        for path in paths:
            observed.extend(json.loads(line)["topic_id"] for line in path.read_text().splitlines() if line.strip())
        if sorted(observed) != sorted(run["tasks"]):
            raise ValueError(f"Finish all native {framework} topics before freezing: got {observed}")
        sources[framework] = paths
    return freeze(sources, destination)
