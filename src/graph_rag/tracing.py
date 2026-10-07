"""Langfuse tracing: the one place that decides what text leaves the process.

Two layers, both deterministic code, neither relying on the model:

1. Content gate. Every ``@observe`` in this package sets ``capture_input=False`` and
   ``capture_output=False``, so the SDK never serialises function arguments or return
   values on its own. Text reaches a span only through :func:`record`, which drops
   ``input``/``output`` unless ``TRACE_CONTENT=full``. ``metadata`` (IDs, scores, counts,
   timings, model name, token usage) is always sent. Any value other than ``full``,
   including a typo, means ``metadata``: the gate fails closed.

2. Mask. :func:`mask` runs inside the Langfuse SDK on every input, output and metadata
   value before export, in both modes. It redacts credential-shaped strings and email
   addresses and truncates long strings. It is defence in depth for ``full`` mode, not a
   PII guarantee: names, addresses and free-form personal data pass through.

Tracing stays inert until LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are set.
"""
from __future__ import annotations

import os
import re
from typing import Any

from langfuse import Langfuse, get_client

FULL = "full"
METADATA = "metadata"
MAX_TRACE_STRING_CHARS = 4000

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED:private-key]"),
    (re.compile(r"\b[sp]k-lf-[0-9a-fA-F-]{8,}"), "[REDACTED:langfuse-key]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "[REDACTED:hf-token]"),
    (re.compile(r"\b(?:sk|gsk|xoxb|xoxp|ghp|gho|github_pat)[-_][A-Za-z0-9_-]{16,}"), "[REDACTED:api-key]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED:aws-key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "[REDACTED:jwt]"),
    (re.compile(r"(?i)\b(bearer|password|passwd|secret|api[_-]?key|token)(\s*[:=]\s*|\s+)(?!\[REDACTED)[^\s,;\"']{6,}"), r"\1\2[REDACTED]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED:email]"),
)

_client: Langfuse | None = None


def content_mode() -> str:
    """``full`` only when explicitly requested; everything else is ``metadata``."""
    return FULL if os.environ.get("TRACE_CONTENT", "").strip().lower() == FULL else METADATA


def redact(data: Any) -> Any:
    if isinstance(data, str):
        for pattern, replacement in _REDACTIONS:
            data = pattern.sub(replacement, data)
        if len(data) > MAX_TRACE_STRING_CHARS:
            data = data[:MAX_TRACE_STRING_CHARS] + f"... [truncated {len(data) - MAX_TRACE_STRING_CHARS} chars]"
        return data
    if isinstance(data, dict):
        return {key: redact(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [redact(value) for value in data]
    return data


def mask(*, data: Any, **_: Any) -> Any:
    """Langfuse ``MaskFunction``: applied by the SDK to input, output and metadata."""
    return redact(data)


def init_tracing(**overrides: Any) -> Langfuse:
    """Create the process-wide Langfuse client with the mask installed.

    Call before the first traced call: the SDK keeps the first client per public key, and
    ``get_client()`` (used by ``@observe``) reuses its settings, mask included. Overrides
    exist for tests (``span_exporter=InMemorySpanExporter()``, fake keys).
    """
    global _client
    if _client is None or overrides:
        _client = Langfuse(mask=mask, **overrides)
    return _client


def reset_for_tests() -> None:
    global _client
    _client = None


def record(*, input: Any = None, output: Any = None, metadata: dict[str, Any] | None = None,
           generation: bool = False, **generation_fields: Any) -> None:
    """Attach data to the current observation, honouring the content gate.

    ``input``/``output`` are content: sent only when TRACE_CONTENT=full.
    ``metadata`` and generation fields (model, usage_details, model_parameters,
    completion_start_time) are sent in both modes, so never put raw text in them.
    """
    fields: dict[str, Any] = {"metadata": {"trace_content": content_mode(), **(metadata or {})}}
    if content_mode() == FULL:
        if input is not None:
            fields["input"] = input
        if output is not None:
            fields["output"] = output
    client = get_client()
    if generation:
        client.update_current_generation(**fields, **generation_fields)
    else:
        client.update_current_span(**fields)


def current_trace_id() -> str:
    return get_client().get_current_trace_id() or ""


def flush() -> None:
    get_client().flush()


def shutdown() -> None:
    get_client().shutdown()
