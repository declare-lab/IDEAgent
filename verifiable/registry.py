"""Task registry used by the shared proposal freeze (proposals.known_task) and the sandbox worker's baseline path.

Program QD registers its own tasks (program_qd/tasks.py) by widening proposals.known_task during its freeze, and runs
only candidate programs in the sandbox, so no benchmark adapters are registered here.
"""
from __future__ import annotations


def provider(task: str):
    raise ValueError(f"Unknown task {task!r}: no benchmark adapters are registered (program QD tasks: program_qd/tasks.py)")


def describe(task: str) -> dict:
    return provider(task)


def known_task(task: str) -> bool:
    return False
