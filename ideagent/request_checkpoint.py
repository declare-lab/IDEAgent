"""Durable successful API responses for deterministic request replay."""
from __future__ import annotations

from collections import defaultdict
import functools
import contextvars
import importlib
import inspect
import copy
import sys
import hashlib
import json
from pathlib import Path
import sqlite3
import threading


class ResumeError(RuntimeError):
    """A checkpointed run cannot start or continue; the message says why and what to do."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# SDK response classes a journal may rebuild, stored by name and resolved through their public
# import path, so a journal can never make resume import a module it names itself.
RESPONSE_TYPES = {
    "openai.ChatCompletion": ("openai.types.chat", "ChatCompletion"),
    "anthropic.Message": ("anthropic.types", "Message"),
    "google.genai.GenerateContentResponse": ("google.genai.types", "GenerateContentResponse"),
}


def _response_type_name(response):
    if isinstance(response, dict):
        return "dict"
    # The SDK that built the response has already imported its public types module.
    for name, (module, attr) in RESPONSE_TYPES.items():
        loaded = sys.modules.get(module)
        if loaded is not None and isinstance(response, getattr(loaded, attr)):
            return name
    raise TypeError(f"Cannot checkpoint {type(response).__module__}.{type(response).__qualname__} responses")


def _changed(old, new, path=()):
    """Dotted paths at which two run identities differ, with their old and new values."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            yield from _changed(old.get(key), new.get(key), (*path, key))
    elif canonical(old) != canonical(new):
        yield f"{'.'.join(map(str, path))} ({old} → {new})"


class RequestJournal:
    def __init__(self, path, identity, *, resume=False):
        import fcntl

        self.path = Path(path)
        if resume and not self.path.exists():
            raise ResumeError(f"{self.path} does not exist, so there is nothing to resume. "
                              "Check --checkpoint-dir, or start afresh without --resume.")
        if not resume and self.path.exists():
            raise ResumeError(f"{self.path} already exists; pass --resume to continue that run, "
                              "or use a new --checkpoint-dir")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._process_lock = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self._process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._process_lock.close()
            raise ResumeError("Another process is using this request checkpoint") from None
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        # WAL keeps its index in a shared-memory file, which network filesystems (NFS, Lustre) do
        # not support. If journals must live on one, run 'PRAGMA locking_mode=EXCLUSIVE' before
        # this line: the flock above already makes this process the journal's only user, and in
        # exclusive mode SQLite keeps the WAL index in heap memory instead.
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS protocol (identity TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT NOT NULL, occurrence INTEGER NOT NULL, request TEXT NOT NULL, response TEXT, PRIMARY KEY(key, occurrence))')
        self.db.execute('CREATE TABLE IF NOT EXISTS responses (key TEXT NOT NULL, occurrence INTEGER NOT NULL, position INTEGER NOT NULL, response TEXT NOT NULL, PRIMARY KEY(key, occurrence, position))')
        previous = self.db.execute('SELECT identity FROM protocol').fetchone()
        expected = canonical(identity)
        if previous and previous[0] != expected:
            self.close()
            changed = ", ".join(_changed(json.loads(previous[0]), json.loads(expected)))
            raise ResumeError(f"Cannot resume: {changed} changed since this run started. Undo the change "
                              "to resume, or start afresh without --resume in a new --checkpoint-dir.")
        if previous is None:
            self.db.execute('INSERT INTO protocol VALUES (?)', (expected,))
        self.db.commit()
        self.lock = threading.Lock()
        self.occurrences = defaultdict(int)
        # A resumed run first rebuilds its unfinished topic from the journal, until the first
        # request it has to send to the API.
        self.replaying = resume

    def reserve(self, request):
        """Claim the next occurrence of a request: its ticket, its finished output if any, and the
        raw-response position at which an unfinished call resumes."""
        serialized = canonical(request)
        key = hashlib.sha256(serialized.encode()).hexdigest()
        with self.lock:
            occurrence = self.occurrences[key]
            self.occurrences[key] += 1
            row = self.db.execute('SELECT response FROM calls WHERE key=? AND occurrence=?', (key, occurrence)).fetchone()
            if row is None:
                self.db.execute('INSERT INTO calls VALUES (?, ?, ?, NULL)', (key, occurrence, serialized))
                self.db.commit()
                return (key, occurrence), None, 0
            if row[0] is not None:
                return (key, occurrence), json.loads(row[0]), 0
            # Every saved response but the last was followed by another send, so it already failed
            # extraction. Resuming at the last one gives the retry loop a fresh budget; replaying
            # them all would re-exhaust it on known failures and crash the same way every resume.
            last = self.db.execute('SELECT MAX(position) FROM responses WHERE key=? AND occurrence=?',
                                   (key, occurrence)).fetchone()[0]
            return (key, occurrence), None, last or 0

    def complete(self, ticket, response):
        with self.lock:
            self.db.execute('UPDATE calls SET response=? WHERE key=? AND occurrence=?', (canonical(response), *ticket))
            self.db.commit()

    def raw_response(self, ticket, position):
        with self.lock:
            row = self.db.execute('SELECT response FROM responses WHERE key=? AND occurrence=? AND position=?', (*ticket, position)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        kind = value.get("type")
        if kind == "dict":
            return value["data"]
        if kind not in RESPONSE_TYPES:
            raise ValueError(f"Checkpoint holds an unsupported response type {kind!r}")
        module, attr = RESPONSE_TYPES[kind]
        return getattr(importlib.import_module(module), attr).model_validate(value["data"])

    def save_response(self, ticket, position, response):
        kind = _response_type_name(response)
        value = {"type": kind, "data": response if kind == "dict" else response.model_dump(mode="json")}
        with self.lock:
            self.db.execute('INSERT INTO responses VALUES (?, ?, ?, ?)', (*ticket, position, canonical(value)))
            self.db.commit()

    def close(self):
        self.db.close()
        self._process_lock.close()


_current = None


def configure(journal):
    global _current
    _current = journal


def current():
    return _current


def replaying():
    """True while a resumed run is rebuilding work that the interrupted run already logged."""
    return _current is not None and _current.replaying


_active_request = contextvars.ContextVar("checkpoint_request", default=None)


def request_response(send):
    """Persist the SDK response before text extraction, including parsing retries."""
    active = _active_request.get()
    if active is None:
        return send()
    journal, ticket, cursor = active
    response = journal.raw_response(ticket, cursor[0])
    if response is None:
        journal.replaying = False
        response = send()
        cursor[1] = False
        journal.save_response(ticket, cursor[0], response)
    cursor[0] += 1
    return response


async def request_response_async(send):
    active = _active_request.get()
    if active is None:
        return await send()
    journal, ticket, cursor = active
    response = journal.raw_response(ticket, cursor[0])
    if response is None:
        journal.replaying = False
        response = await send()
        cursor[1] = False
        journal.save_response(ticket, cursor[0], response)
    cursor[0] += 1
    return response


def checkpointed(fn):
    """Journal one backend request, outside its existing transient-error retry loop."""
    def prepare(client, messages, kwargs):
        journal = current()
        if journal is None:
            return None, None, None, 0
        ticket, response, start = journal.reserve({
            "backend": type(client).__name__, "model": client.model_id,
            "role": getattr(client, "checkpoint_role", ""),
            "method": fn.__name__, "messages": messages, "kwargs": kwargs,
        })
        if response is not None:
            client._set_response_metadata({**(response["metadata"] or {}), "checkpoint_replay": True})
        return journal, ticket, response, start

    def complete(client, journal, ticket, output):
        if journal is not None:
            active = _active_request.get()
            if active is not None and active[2][0] and active[2][1]:
                client._set_response_metadata({**(client.get_last_response_metadata() or {}), "checkpoint_replay": True})
            journal.complete(ticket, {"output": output, "metadata": client.get_last_response_metadata()})
        return output

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_call(client, messages, **kwargs):
            journal, ticket, response, start = prepare(client, messages, kwargs)
            if response is not None:
                return response["output"]
            token = _active_request.set((journal, ticket, [start, True]) if journal is not None else None)
            try:
                output = await fn(client, messages, **kwargs)
                return complete(client, journal, ticket, output)
            finally:
                _active_request.reset(token)
        return async_call

    @functools.wraps(fn)
    def call(client, messages, **kwargs):
        journal, ticket, response, start = prepare(client, messages, kwargs)
        if response is not None:
            return response["output"]
        token = _active_request.set((journal, ticket, [start, True]) if journal is not None else None)
        try:
            output = fn(client, messages, **kwargs)
            return complete(client, journal, ticket, output)
        finally:
            _active_request.reset(token)
    return call


# Per-role settings that change how a request is authenticated or routed, not what it asks.
OPERATIONAL_AGENT_KEYS = ("api_key_env", "port")


class ReplayAwareLogger:
    """Wraps a run's trace or response logger so that a resumed run logs nothing twice.

    Nothing is written while the unfinished topic is rebuilt from the journal. During that time
    a model call's start is held until the call ends, and written only if the call reached the
    API. A call answered from the journal is never written: the interrupted run logged it.
    """

    def __init__(self, logger):
        self._logger = logger
        self._held_starts = {}

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def __enter__(self):
        self._logger.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._logger.__exit__(*exc_info)

    def log(self, *args, **kwargs):
        event = args[0] if args else kwargs.get("event")
        if event == "model_call_started" and replaying():
            self._held_starts[kwargs["call_id"]] = (args, kwargs)
            return None
        if event in ("model_call_completed", "model_call_failed"):
            start = self._held_starts.pop(kwargs["call_id"], None)
            if (kwargs.get("response_metadata") or {}).get("checkpoint_replay"):
                return None
            if start is not None and not replaying():
                self._logger.log(*start[0], **start[1])
        return None if replaying() else self._logger.log(*args, **kwargs)


def replay_aware(logger):
    """The logger a run should use: wrapped only when a request journal is active."""
    return logger if _current is None else ReplayAwareLogger(logger)


def prepare_checkpoint(config, root, *, resume=False):
    """Open the run's request journal; a resumed run continues in the same output directory."""
    # Resume refuses before anything runs if a role's model or generation settings changed.
    # Prompts (prompt code, the input file) are not checked: a request whose prompt changed no
    # longer matches the journal and is sent fresh. Provider endpoints may change too: saved
    # answers are replayed whichever endpoint gave them.
    identity = {
        "version": 2,
        "agents": {role: {key: value for key, value in (agent or {}).items() if key not in OPERATIONAL_AGENT_KEYS}
                   for role, agent in (config.get("agents") or {}).items()},
    }
    journal = RequestJournal(Path(root) / "requests.sqlite", identity, resume=resume)
    if resume:
        # Skip the topics the interrupted run finished and keep its outputs, as run.resume does;
        # its unfinished topic is rebuilt from the journal.
        config = copy.deepcopy(config)
        config.setdefault("run", {})["resume"] = True
    return journal, config
