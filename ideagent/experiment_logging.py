"""Thread-safe, append-only provenance logging for generation experiments.

The trace stores complete prompts and model outputs locally for reproducibility, but never client
objects, environment variables, or API keys.  A run id plus per-process sequence number makes
records unambiguous even when a resumed experiment appends to an existing JSONL file.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


_SENSITIVE_KEYS = {
    "api_key", "apikey", "authorization", "password", "secret", "token",
    "openai_api_key", "gemini_api_key",
}


def _safe_json(value: Any) -> Any:
    """Make call metadata serializable and redact fields that could contain credentials."""

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            out[str(key)] = "[REDACTED]" if normalized in _SENSITIVE_KEYS else _safe_json(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ExperimentLogger:
    """Synchronous append logger safe for concurrent soundness-panel calls."""

    def __init__(self, path: str | Path, *, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._sequence = 0

    def log(self, event: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "sequence": self._sequence,
                "event": event,
                **_safe_json(payload),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_run_manifest(
    output_dir: str | Path, *, logger: ExperimentLogger, manifest: dict[str, Any]
) -> Path:
    """Write an immutable per-run protocol record beside the append-only event trace."""

    path = Path(output_dir) / f"run_manifest_{logger.run_id}.json"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": logger.run_id,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "working_directory": os.getcwd(),
        },
        **_safe_json(manifest),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def _usage_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if value.get("checkpoint_replay"):
            return
        usage = value.get("usage")
        if isinstance(usage, dict):
            yield usage
        for key, item in value.items():
            if key != "usage":
                yield from _usage_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _usage_records(item)


def _number(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _normalized_usage(usage: dict[str, Any]) -> dict[str, int]:
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    return {
        "input_tokens": _number(usage, "prompt_tokens", "prompt_token_count", "input_tokens"),
        "output_tokens": _number(
            usage, "completion_tokens", "candidates_token_count", "output_tokens"
        ),
        "total_tokens": _number(usage, "total_tokens", "total_token_count"),
        "cached_input_tokens": (
            _number(usage, "cached_content_token_count")
            + (_number(prompt_details, "cached_tokens") if isinstance(prompt_details, dict) else 0)
        ),
        "reasoning_tokens": (
            _number(usage, "thoughts_token_count", "thought_token_count")
            + (_number(completion_details, "reasoning_tokens") if isinstance(completion_details, dict) else 0)
        ),
    }


def summarize_experiment_trace(
    path: str | Path, *, run_id: str | None = None
) -> dict[str, Any]:
    """Aggregate calls, latency, and provider token fields by role and stage."""

    trace_path = Path(path)
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    totals = {
        "calls_started": 0, "calls_completed": 0, "calls_failed": 0,
        "latency_seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
        "total_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0,
    }
    if not trace_path.exists():
        return {"run_id": run_id, "totals": totals, "by_role_stage_model": []}

    with trace_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if run_id is not None:
        rows = [row for row in rows if row.get("run_id") == run_id]

    for row in rows:
        event = row.get("event")
        if event not in {"model_call_started", "model_call_completed", "model_call_failed"}:
            continue
        role = str(row.get("role", "unknown"))
        stage = str((row.get("context") or {}).get("stage", "unspecified"))
        model_id = str(row.get("model_id", ""))
        key = (role, stage, model_id)
        group = groups.setdefault(key, {
            "role": role, "stage": stage, "model_id": model_id,
            "calls_started": 0, "calls_completed": 0, "calls_failed": 0,
            "latency_seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0,
        })
        count_key = {
            "model_call_started": "calls_started",
            "model_call_completed": "calls_completed",
            "model_call_failed": "calls_failed",
        }[event]
        group[count_key] += 1
        totals[count_key] += 1
        if event in {"model_call_completed", "model_call_failed"}:
            latency = float(row.get("latency_seconds", 0.0) or 0.0)
            group["latency_seconds"] += latency
            totals["latency_seconds"] += latency
        if event == "model_call_completed":
            for usage in _usage_records(row.get("response_metadata")):
                normalized = _normalized_usage(usage)
                for token_key, value in normalized.items():
                    group[token_key] += value
                    totals[token_key] += value

    for target in [totals, *groups.values()]:
        target["latency_seconds"] = round(float(target["latency_seconds"]), 6)
    return {
        "run_id": run_id,
        "trace_path": str(trace_path),
        "totals": totals,
        "by_role_stage_model": sorted(
            groups.values(), key=lambda item: (item["role"], item["stage"], item["model_id"])
        ),
    }


def write_trace_summary(output_dir: str | Path, *, logger: ExperimentLogger) -> Path:
    summary = summarize_experiment_trace(logger.path, run_id=logger.run_id)
    path = Path(output_dir) / f"trace_summary_{logger.run_id}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class TracedClient:
    """Transparent LLM client wrapper that logs every complete prompt, output, error, and latency."""

    def __init__(self, client: Any, *, logger: ExperimentLogger, role: str) -> None:
        self._client = client
        self.trace_logger = logger
        self.trace_role = role
        self.model_id = getattr(client, "model_id", "")
        self._context_lock = threading.Lock()
        self._context: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def set_trace_context(self, **context: Any) -> None:
        with self._context_lock:
            self._context = dict(context)

    def _context_copy(self) -> dict[str, Any]:
        with self._context_lock:
            return dict(self._context)

    def _start(self, *, method: str, messages: Any, kwargs: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
        call_id = uuid.uuid4().hex
        context = self._context_copy()
        self.trace_logger.log(
            "model_call_started",
            call_id=call_id,
            role=self.trace_role,
            model_id=self.model_id,
            method=method,
            context=context,
            messages=messages,
            request_kwargs=kwargs,
        )
        return call_id, time.perf_counter(), context

    def _complete(
        self, *, call_id: str, started: float, method: str, context: dict[str, Any], output: Any
    ) -> None:
        metadata_getter = getattr(self._client, "get_last_response_metadata", None)
        response_metadata = metadata_getter() if callable(metadata_getter) else None
        self.trace_logger.log(
            "model_call_completed",
            call_id=call_id,
            role=self.trace_role,
            model_id=self.model_id,
            method=method,
            context=context,
            latency_seconds=round(time.perf_counter() - started, 6),
            response_metadata=response_metadata,
            output=output,
        )

    def _failed(
        self, *, call_id: str, started: float, method: str, context: dict[str, Any], exc: Exception
    ) -> None:
        self.trace_logger.log(
            "model_call_failed",
            call_id=call_id,
            role=self.trace_role,
            model_id=self.model_id,
            method=method,
            context=context,
            latency_seconds=round(time.perf_counter() - started, 6),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        call_id, started, context = self._start(method="generate", messages=messages, kwargs=kwargs)
        try:
            output = self._client.generate(messages, **kwargs)
        except Exception as exc:
            self._failed(
                call_id=call_id, started=started, method="generate", context=context, exc=exc
            )
            raise
        self._complete(
            call_id=call_id, started=started, method="generate", context=context, output=output
        )
        return output

    def generate_many(
        self, messages: list[dict[str, str]], *, n: int = 1, **kwargs: Any
    ) -> list[str]:
        request_kwargs = {**kwargs, "n": n}
        call_id, started, context = self._start(
            method="generate_many", messages=messages, kwargs=request_kwargs
        )
        try:
            output = self._client.generate_many(messages, n=n, **kwargs)
        except Exception as exc:
            self._failed(
                call_id=call_id, started=started, method="generate_many", context=context, exc=exc
            )
            raise
        self._complete(
            call_id=call_id, started=started, method="generate_many", context=context, output=output
        )
        return output

    async def generate_async(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        call_id, started, context = self._start(
            method="generate_async", messages=messages, kwargs=kwargs
        )
        try:
            if hasattr(self._client, "generate_async"):
                output = await self._client.generate_async(messages, **kwargs)
            else:
                output = await asyncio.to_thread(self._client.generate, messages, **kwargs)
        except Exception as exc:
            self._failed(
                call_id=call_id, started=started, method="generate_async", context=context, exc=exc
            )
            raise
        self._complete(
            call_id=call_id, started=started, method="generate_async", context=context, output=output
        )
        return output


def set_trace_context(client: Any, **context: Any) -> None:
    setter = getattr(client, "set_trace_context", None)
    if callable(setter):
        setter(**context)
