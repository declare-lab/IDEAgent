"""LLM clients for the agentic ideation system.

vLLM (OpenAI-compatible, local server), Gemini (google-genai), OpenAI-hosted (real
api.openai.com), and Claude (anthropic's direct Anthropic API client) backends.
``build_client`` routes any model whose name starts with ``gemini`` or ``gemma`` to the
Gemini backend (its port is ignored) -- Gemma models are served through the same
google.genai.Client as Gemini -- any model starting with ``gpt-`` to the OpenAI backend
(e.g. gpt-5.6-sol), and any model starting with ``claude`` to the Claude backend (e.g.
claude-opus-4-7; its port is ignored -- auth is the ANTHROPIC_API_KEY environment
variable, read by the SDK directly).

Topics still run strictly sequentially so each topic's learnings are made and persisted
before the next begins. The learning loop may overlap independent role memory updates.
"""
from __future__ import annotations

import asyncio
import backoff
import contextvars
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

from ideagent.request_checkpoint import checkpointed, request_response, request_response_async

MAX_TRIES = 5


def _jsonable(value: Any) -> Any:
    """Best-effort conversion for provider response metadata (usage, ids, finish reasons)."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(exclude_none=True))
        except TypeError:
            return _jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return str(value)


class _ResponseMetadataMixin:
    """Expose metadata for the just-finished call without cross-thread panel races."""

    def _init_response_metadata(self) -> None:
        self._response_metadata_var: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar(f"response_metadata_{id(self)}", default=None)
        )

    def _set_response_metadata(self, metadata: dict[str, Any]) -> None:
        if not hasattr(self, "_response_metadata_var"):
            self._init_response_metadata()
        self._response_metadata_var.set(_jsonable(metadata))

    def get_last_response_metadata(self) -> dict[str, Any] | None:
        if not hasattr(self, "_response_metadata_var"):
            return None
        return self._response_metadata_var.get()


def _completion_metadata(completion: Any) -> dict[str, Any]:
    """Capture reproducibility/cost fields common to OpenAI-compatible responses."""

    choices = getattr(completion, "choices", None)
    if choices is None and isinstance(completion, dict):
        choices = completion.get("choices")
    finish_reasons = []
    for choice in choices or []:
        reason = getattr(choice, "finish_reason", None)
        if reason is None and isinstance(choice, dict):
            reason = choice.get("finish_reason")
        finish_reasons.append(_jsonable(reason))
    return {
        "response_id": _jsonable(getattr(completion, "id", None)),
        "response_model": _jsonable(getattr(completion, "model", None)),
        "created": _jsonable(getattr(completion, "created", None)),
        "system_fingerprint": _jsonable(getattr(completion, "system_fingerprint", None)),
        "usage": _jsonable(getattr(completion, "usage", None)),
        "finish_reasons": finish_reasons,
    }


def _retry(fn: Callable) -> Callable:
    """Retry a client call on ANY exception (timeouts, transient generation errors), up to
    ``MAX_TRIES`` with exponential backoff."""
    return backoff.on_exception(backoff.expo, Exception, max_tries=MAX_TRIES)(fn)


def _async_retry(fn: Callable) -> Callable:
    """Async version of ``_retry`` — backoff waits with ``asyncio.sleep``, so it never blocks
    the event loop."""
    return backoff.on_exception(backoff.expo, Exception, max_tries=MAX_TRIES)(fn)


class LLMClient(Protocol):
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...


class VLLMOpenAIClient(_ResponseMetadataMixin):
    """vLLM client using the OpenAI-compatible Python client."""

    def __init__(
        self,
        *,
        model_id: str | None,
        port: int = 8000,
        api_key_env: str = "VLLM_API_KEY",
        timeout: float = 120.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The vLLM backend requires the `openai` Python package. "
                "Install it in the project environment, e.g. `uv pip install openai`."
            ) from exc
        self.port = int(port)
        self.base_url = f"http://localhost:{self.port}/v1"
        api_key = os.environ.get(api_key_env, "EMPTY")
        self.client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout)
        if not model_id:
            raise ValueError("Pass a model id for the vLLM backend.")
        self.model_id = model_id
        self._init_response_metadata()

    def _chat_completion_create(self, **kwargs: Any):
        chat = getattr(self.client, "chat", None)
        if chat is not None and hasattr(chat, "completions"):
            return chat.completions.create(**kwargs)
        chat_completions = getattr(self.client, "chat_completions", None)
        if chat_completions is not None:
            if hasattr(chat_completions, "create"):
                return chat_completions.create(**kwargs)
            return chat_completions(**kwargs)
        raise AttributeError("OpenAI client does not expose a chat completions API")

    def _chat_request(self, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> dict[str, Any]:
        extra_body: dict[str, Any] = {}
        # enable_thinking: True/False sets it; None => omit entirely (model may not support it).
        enable_thinking = kwargs.get("enable_thinking", True)
        if enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        top_k = kwargs.get("top_k")
        if top_k is not None and int(top_k) > 0:
            extra_body["top_k"] = int(top_k)
        min_p = kwargs.get("min_p")
        if min_p is not None:
            extra_body["min_p"] = float(min_p)
        repetition_penalty = kwargs.get("repetition_penalty")
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = float(repetition_penalty)
        request: dict[str, Any] = {
            "model": kwargs.get("model_id") or self.model_id,
            "messages": messages,
            "temperature": float(kwargs.get("temperature", 0.7)),
            "top_p": float(kwargs.get("top_p", 0.95)),
            "max_completion_tokens": int(kwargs.get("max_new_tokens", 8192)),
        }
        presence_penalty = kwargs.get("presence_penalty")
        if presence_penalty is not None:
            request["presence_penalty"] = float(presence_penalty)
        response_format = kwargs.get("response_format")
        if response_format is not None:
            request["response_format"] = response_format
        if extra_body:
            request["extra_body"] = extra_body
        return request

    @staticmethod
    def _choice_content(choice: Any) -> str:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("vLLM response choice did not contain message content")
        return content

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self.generate_many(messages, n=1, **kwargs)[0]

    @checkpointed
    @_retry
    def generate_many(self, messages: list[dict[str, str]], *, n: int = 1, **kwargs: Any) -> list[str]:
        if int(n) <= 0:
            raise ValueError("n must be a positive integer")
        request = self._chat_request(messages, kwargs)
        if int(n) != 1:
            request["n"] = int(n)
        completion = request_response(lambda: self._chat_completion_create(**request))
        self._set_response_metadata(_completion_metadata(completion))
        choices = getattr(completion, "choices", None)
        if choices is None and isinstance(completion, dict):
            choices = completion.get("choices")
        if not choices:
            raise ValueError("vLLM response did not contain choices")
        outputs = [self._choice_content(choice) for choice in choices]
        if len(outputs) < int(n):
            raise ValueError(f"vLLM returned {len(outputs)} choices for n={int(n)}")
        return outputs

    async def generate_async(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        # Reuses the sync generate (which already has retry) via asyncio.to_thread so the
        # event loop is not blocked during the HTTP round-trip.
        return await asyncio.to_thread(self.generate, messages, **kwargs)


def _build_gemini_config(kwargs: dict[str, Any]) -> Any:
    """Shared helper for sync and async Gemini generate methods."""
    from google.genai import types

    system_instruction: str | None = None
    turns = []
    for msg in kwargs.get("_messages", []):
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_instruction = content
        elif role == "user":
            turns.append(types.Content(role="user", parts=[types.Part(text=content)]))
        elif role == "assistant":
            turns.append(types.Content(role="model", parts=[types.Part(text=content)]))

    cfg = types.GenerateContentConfig(
        temperature=float(kwargs.get("temperature", 0.7)),
        top_p=float(kwargs.get("top_p", 0.95)),
        max_output_tokens=int(kwargs.get("max_new_tokens", 8192)),
    )
    if system_instruction:
        cfg.system_instruction = system_instruction
    top_k = kwargs.get("top_k")
    if top_k is not None and int(top_k) > 0:
        cfg.top_k = int(top_k)
    response_format = kwargs.get("response_format")
    if response_format is not None:
        cfg.response_mime_type = "application/json"
        schema = (response_format.get("json_schema") or {}).get("schema")
        if schema:
            # response_json_schema accepts raw JSON Schema dicts directly and correctly
            # handles additionalProperties, required, enum, etc. response_schema would
            # attempt a lossy conversion through the SDK's internal Schema type.
            cfg.response_json_schema = schema

    # Thinking-control mechanism is model-generation-specific and mutually exclusive:
    # Gemini 2.x models only accept thinking_budget and reject thinking_level outright
    # ("Thinking level is not supported for this model."); Gemma 4 and Gemini 3.x only
    # accept thinking_level and reject/ignore thinking_budget. Setting both in one config
    # is also rejected, so exactly one path below must run.
    model_id = str(kwargs.get("_model_id", "")).lower()
    uses_thinking_budget = "gemini-2" in model_id

    enable_thinking = kwargs.get("enable_thinking")
    if enable_thinking is False:
        if uses_thinking_budget:
            cfg.thinking_config = types.ThinkingConfig(thinking_budget=0)
        else:
            # MINIMAL is the "disabled" equivalent for Gemma 4 / Gemini 3.x.
            cfg.thinking_config = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
    elif enable_thinking is True:
        thinking_effort = kwargs.get("thinking_effort")
        if uses_thinking_budget:
            # No thinking_level support on Gemini 2.x, and thinking_effort ("high"/"low")
            # isn't a token budget -- leave thinking_config unset so the model uses its
            # own default/dynamic thinking budget rather than guessing a token count.
            pass
        elif thinking_effort is not None:
            level = getattr(types.ThinkingLevel, thinking_effort.upper(), None)
            cfg.thinking_config = types.ThinkingConfig(
                thinking_level=level, include_thoughts=False
            )
        else:
            cfg.thinking_config = types.ThinkingConfig(include_thoughts=False)

    return turns, cfg


def _extract_gemini_text(response: Any) -> str:
    parts = response.candidates[0].content.parts
    text = "".join(
        p.text for p in parts
        if p.text and not getattr(p, "thought", False)
    )
    if not text:
        raise ValueError("Gemini response contained no non-thought text content")
    return text


class GeminiClient(_ResponseMetadataMixin):
    """Google Generative AI client for Gemini models (port is irrelevant)."""

    def __init__(self, *, model_id: str, api_key_env: str = "GEMINI_API_KEY", timeout: float = 120.0) -> None:
        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The Gemini backend requires the `google-genai` package. "
                "Install it with: `uv pip install google-genai`."
            ) from exc
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ModuleNotFoundError:
            pass  # dotenv optional; fall back to plain os.environ
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"Gemini API key not found. Set the {api_key_env!r} environment variable "
                "or add it to a .env file."
            )
        self.model_id = model_id
        # timeout in HttpOptions is milliseconds
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )
        self._genai = genai
        self._init_response_metadata()

    @checkpointed
    @_retry
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        turns, cfg = _build_gemini_config({**kwargs, "_messages": messages, "_model_id": self.model_id})
        response = request_response(lambda: self.client.models.generate_content(
            model=self.model_id, contents=turns, config=cfg
        ))
        self._set_response_metadata({
            "response_id": getattr(response, "response_id", None),
            "response_model": getattr(response, "model_version", None),
            "usage": getattr(response, "usage_metadata", None),
            "finish_reasons": [
                getattr(candidate, "finish_reason", None)
                for candidate in (getattr(response, "candidates", None) or [])
            ],
        })
        return _extract_gemini_text(response)

    @checkpointed
    @_async_retry
    async def generate_async(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        turns, cfg = _build_gemini_config({**kwargs, "_messages": messages, "_model_id": self.model_id})
        response = await request_response_async(lambda: self.client.aio.models.generate_content(
            model=self.model_id, contents=turns, config=cfg
        ))
        self._set_response_metadata({
            "response_id": getattr(response, "response_id", None),
            "response_model": getattr(response, "model_version", None),
            "usage": getattr(response, "usage_metadata", None),
            "finish_reasons": [
                getattr(candidate, "finish_reason", None)
                for candidate in (getattr(response, "candidates", None) or [])
            ],
        })
        return _extract_gemini_text(response)

    def generate_many(self, messages: list[dict[str, str]], *, n: int = 1, **kwargs: Any) -> list[str]:
        """Gemini has no native multi-candidate request, so fan out n independent
        generate() calls (each already @_retry'd) in parallel instead."""
        def one(index) -> tuple[str, dict[str, Any] | None]:
            child_kwargs = {**kwargs, "_checkpoint_slot": [kwargs.get("_checkpoint_slot"), index]}
            output = self.generate(messages, **child_kwargs)
            return output, self.get_last_response_metadata()

        with ThreadPoolExecutor(max_workers=max(1, n)) as executor:
            futures = [executor.submit(one, index) for index in range(n)]
            results = [future.result() for future in futures]
        self._set_response_metadata({
            "fanout_calls": n,
            "subcall_response_metadata": [metadata for _, metadata in results],
        })
        return [output for output, _ in results]


class OpenAIClient(_ResponseMetadataMixin):
    """Real OpenAI-hosted models (e.g. gpt-5.6-sol) via the standard Chat Completions API.

    Separate from VLLMOpenAIClient (rather than reusing it with a base_url override)
    because that client's extra_body fields -- chat_template_kwargs, top_k, min_p,
    repetition_penalty -- are vLLM/SGLang-specific sampling knobs the real OpenAI API
    does not accept; reasoning effort is controlled by the top-level reasoning_effort
    field instead (low/medium/high), reusing enable_thinking/thinking_effort from the
    same yaml gen_kwargs convention used for Gemini's thinking_effort.
    """

    def __init__(
        self,
        *,
        model_id: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The OpenAI backend requires the `openai` Python package. "
                "Install it in the project environment, e.g. `uv pip install openai`."
            ) from exc
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ModuleNotFoundError:
            pass  # dotenv optional; fall back to plain os.environ
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"OpenAI API key not found. Set the {api_key_env!r} environment variable "
                "or add it to a .env file."
            )
        self.model_id = model_id
        # No base_url override -- the OpenAI SDK's own default is https://api.openai.com/v1.
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self._init_response_metadata()

    def _chat_request(self, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> dict[str, Any]:
        # temperature/top_p/presence_penalty/frequency_penalty are deliberately omitted --
        # OpenAI's reasoning models (gpt-5.6-sol and the rest of the GPT-5.x reasoning
        # family) reject non-default values for these outright via Chat Completions
        # ("Unsupported value: 'temperature' does not support 0.2 with this model"),
        # since sampling is tuned around the reasoning process itself; reasoning_effort is
        # the supported steering knob instead.
        request: dict[str, Any] = {
            "model": kwargs.get("model_id") or self.model_id,
            "messages": messages,
            "max_completion_tokens": int(kwargs.get("max_new_tokens", 8192)),
        }
        if kwargs.get("enable_thinking"):
            thinking_effort = kwargs.get("thinking_effort")
            if thinking_effort:
                request["reasoning_effort"] = thinking_effort
        # On by default -- the background/avoidance context resent every single generation
        # here is a stable, repeated prefix (see clients.py module docstring), exactly what
        # OpenAI's prompt caching is for. prompt_cache_key (set by the caller) is what makes
        # matching reliable on GPT-5.6+ per OpenAI's own guidance -- without it caching
        # still works but is best-effort only. NOTE: prompt_cache_options (mode/ttl) does
        # not exist on the installed SDK's Completions.create() -- only prompt_cache_key is
        # actually supported.
        if kwargs.get("prompt_caching", True):
            cache_key = kwargs.get("prompt_cache_key")
            if cache_key:
                request["prompt_cache_key"] = cache_key
        response_format = kwargs.get("response_format")
        if response_format is not None:
            request["response_format"] = response_format
        return request

    @staticmethod
    def _choice_content(choice: Any) -> str:
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenAI response choice did not contain message content")
        return content

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self.generate_many(messages, n=1, **kwargs)[0]

    @checkpointed
    @_retry
    def generate_many(self, messages: list[dict[str, str]], *, n: int = 1, **kwargs: Any) -> list[str]:
        if int(n) <= 0:
            raise ValueError("n must be a positive integer")
        request = self._chat_request(messages, kwargs)
        if int(n) != 1:
            request["n"] = int(n)
        completion = request_response(lambda: self.client.chat.completions.create(**request))
        self._set_response_metadata(_completion_metadata(completion))
        outputs = [self._choice_content(choice) for choice in completion.choices]
        if len(outputs) < int(n):
            raise ValueError(f"OpenAI returned {len(outputs)} choices for n={int(n)}")
        return outputs

    async def generate_async(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return await asyncio.to_thread(self.generate, messages, **kwargs)


def _build_anthropic_request(
    model_id: str, messages: list[dict[str, str]], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Shared helper for sync and async Claude generate methods."""
    system: str | None = None
    turns = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            turns.append({"role": msg["role"], "content": msg["content"]})

    response_format = kwargs.get("response_format")
    if response_format is not None and turns and turns[-1]["role"] == "user":
        # No server-side schema enforcement is sent: the schema is shown to the model
        # directly as prompt text instead -- appended to the last user turn rather than
        # the system prompt, since a trailing instruction is followed more reliably than
        # one buried earlier in a long system prompt.
        schema = (response_format.get("json_schema") or {}).get("schema")
        if schema:
            turns[-1] = {
                **turns[-1],
                "content": (
                    turns[-1]["content"]
                    + "\n\nRespond with a single JSON object matching exactly this JSON "
                    "Schema (required keys, types, and any enum values are binding):\n"
                    + json.dumps(schema, indent=2)
                    + "\nOutput the JSON object only. No markdown code fences, no text "
                    "before or after it."
                ),
            }

    request: dict[str, Any] = {
        "model": model_id,
        "messages": turns,
        "max_tokens": int(kwargs.get("max_new_tokens", 8192)),
    }
    if system:
        request["system"] = system

    # Claude Opus 4.7 (and the rest of the adaptive-only Claude 5-era family) rejects
    # temperature/top_p/top_k outright with a 400 -- sampling is tuned around adaptive
    # thinking/effort instead, mirroring the OpenAI reasoning-model handling above.
    # thinking:{type:adaptive} must be set explicitly to turn thinking on at all; effort
    # (below) then controls how much, and also shapes plain non-thinking output/tool calls.
    if kwargs.get("enable_thinking"):
        request["thinking"] = {"type": "adaptive"}

    output_config: dict[str, Any] = {}
    thinking_effort = kwargs.get("thinking_effort")
    if thinking_effort:
        output_config["effort"] = thinking_effort
    # output_config.format (real structured-output enforcement) is deliberately NOT sent.
    # The schema is shown as plain prompt text instead, and the parsers already retry on
    # malformed JSON (max_parsing_retries) -- that pre-dates schema enforcement being
    # added for OpenAI/Gemini, so Claude leans on both of those as its fallback.
    if output_config:
        request["output_config"] = output_config

    return request


def _extract_anthropic_text(message: Any) -> str:
    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    if not text:
        raise ValueError("Claude response contained no text content")
    return text


class AnthropicClient(_ResponseMetadataMixin):
    """Claude models via the `anthropic` SDK's direct Anthropic API client.

    Auth is an API key (ANTHROPIC_API_KEY by default, or api_key_env if that yaml key is
    set) -- vllm_port from build_client() is accepted for call-site signature uniformity
    but unused here, the same way GeminiClient ignores vllm_port.
    """

    def __init__(
        self, *, model_id: str, api_key_env: str = "ANTHROPIC_API_KEY", timeout: float = 120.0
    ) -> None:
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The Claude backend requires the `anthropic` package. Install it with: "
                "`uv pip install anthropic`."
            ) from exc
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ModuleNotFoundError:
            pass  # dotenv optional; fall back to plain os.environ
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} not found. Set it in the environment or a .env file."
            )
        self.model_id = model_id
        self.client = Anthropic(api_key=api_key, timeout=timeout)
        self.async_client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self._init_response_metadata()

    def _set_metadata_from_message(self, message: Any) -> None:
        self._set_response_metadata({
            "response_id": getattr(message, "id", None),
            "response_model": getattr(message, "model", None),
            "usage": getattr(message, "usage", None),
            "finish_reasons": [getattr(message, "stop_reason", None)],
        })

    @checkpointed
    @_retry
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        request = _build_anthropic_request(self.model_id, messages, kwargs)
        message = request_response(lambda: self.client.messages.create(**request))
        self._set_metadata_from_message(message)
        return _extract_anthropic_text(message)

    @checkpointed
    @_async_retry
    async def generate_async(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        request = _build_anthropic_request(self.model_id, messages, kwargs)
        message = await request_response_async(lambda: self.async_client.messages.create(**request))
        self._set_metadata_from_message(message)
        return _extract_anthropic_text(message)

    def generate_many(self, messages: list[dict[str, str]], *, n: int = 1, **kwargs: Any) -> list[str]:
        """Claude has no native multi-candidate request, so fan out n independent
        generate() calls (each already @_retry'd) in parallel, matching GeminiClient."""
        def one(index) -> tuple[str, dict[str, Any] | None]:
            child_kwargs = {**kwargs, "_checkpoint_slot": [kwargs.get("_checkpoint_slot"), index]}
            output = self.generate(messages, **child_kwargs)
            return output, self.get_last_response_metadata()

        with ThreadPoolExecutor(max_workers=max(1, n)) as executor:
            futures = [executor.submit(one, index) for index in range(n)]
            results = [future.result() for future in futures]
        self._set_response_metadata({
            "fanout_calls": n,
            "subcall_response_metadata": [metadata for _, metadata in results],
        })
        return [output for output, _ in results]


def build_client(
    *,
    model_id: str | None = None,
    vllm_port: int = 8000,
    api_key_env: str = "VLLM_API_KEY",
    request_timeout: float = 120.0,
) -> LLMClient:
    if model_id and model_id.lower().startswith(("gemini", "gemma")):
        return GeminiClient(model_id=model_id, api_key_env=api_key_env, timeout=request_timeout)
    if model_id and model_id.lower().startswith("gpt-"):
        return OpenAIClient(model_id=model_id, api_key_env=api_key_env, timeout=request_timeout)
    if model_id and model_id.lower().startswith("claude"):
        claude_api_key_env = "ANTHROPIC_API_KEY" if api_key_env == "VLLM_API_KEY" else api_key_env
        return AnthropicClient(
            model_id=model_id, api_key_env=claude_api_key_env, timeout=request_timeout
        )
    return VLLMOpenAIClient(
        model_id=model_id,
        port=vllm_port,
        api_key_env=api_key_env,
        timeout=request_timeout,
    )
