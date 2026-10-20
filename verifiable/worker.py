"""Private child entrypoint: import code, call solve, emit solution, never score it."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
from time import perf_counter as _trusted_clock
import traceback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--task", required=True)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--cpu-seconds", type=int, default=12)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (args.memory_mb * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds, args.cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    payload = json.loads(sys.stdin.read())
    try:
        # Candidate logging goes to the diagnostic stream, not the JSON envelope.
        with contextlib.redirect_stdout(sys.stderr):
            if args.baseline:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from verifiable.registry import provider
                adapter = provider(args.task)
                solve = lambda problem: adapter.baseline(args.task, problem)
            else:
                code = Path(args.code).resolve()
                sys.path.insert(0, str(code.parent))
                spec = importlib.util.spec_from_file_location("candidate", code)
                if spec is None or spec.loader is None:
                    raise ValueError("Cannot load candidate")
                module = importlib.util.module_from_spec(spec)
                sys.modules["candidate"] = module
                spec.loader.exec_module(module)
                solve = getattr(module, "solve", None)
                if not callable(solve):
                    raise ValueError("Candidate must define solve(problem)")
            started = _trusted_clock()
            solution = solve(payload["problem"])
            elapsed = _trusted_clock() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid measured runtime")
        print(json.dumps({"status": "ok", "solution": solution, "solve_seconds": elapsed}, allow_nan=False))
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:2000]}"}))


if __name__ == "__main__":
    main()
