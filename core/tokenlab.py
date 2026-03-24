"""Token counting, cost estimation, latency measurement, and budget checking.

Reusable across classes 3, 5, 14-16 of the AI Engineer Bootcamp.
Supports Gemini and Groq providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Pricing:
    input_per_1k: float
    output_per_1k: float
    currency: str = "USD"


@dataclass
class LatencyResult:
    ttft_s: float | None
    total_s: float
    tps: float | None
    input_tokens: int | None
    output_tokens: int | None
    output_text: str
    meta: dict = field(default_factory=dict)


class BudgetExceededError(RuntimeError):
    """Raised when a request exceeds the configured budget."""


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def _get_gemini_client():
    """Return (genai.Client, model_name) for Gemini."""
    from google import genai
    client = genai.Client()
    model = get_settings().llm_model
    return client, model


def _get_groq_client():
    """Return (Groq client, model_name) for Groq."""
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY environment variable.")
    client = Groq(api_key=api_key)
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return client, model


def _resolve_provider(provider: str | None) -> str:
    return (provider or get_settings().llm_provider).lower()


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

def count_tokens(text: str, provider: str | None = None) -> int:
    """Count tokens in *text*. Gemini uses the native API; Groq approximates."""
    provider = _resolve_provider(provider)

    if provider == "gemini":
        client, model = _get_gemini_client()
        result = client.models.count_tokens(model=model, contents=text)
        if hasattr(result, "total_tokens"):
            return int(result.total_tokens)
        try:
            return int(result)
        except (TypeError, ValueError):
            raise RuntimeError(f"Cannot extract token count from {result!r}")

    if provider == "groq":
        # Groq has no count_tokens endpoint.
        # Use tiktoken if available; otherwise ≈ 4 chars/token.
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return max(1, len(text) // 4)

    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: Pricing | None = None,
) -> float:
    """Estimate cost from token counts and pricing. Returns 0.0 when pricing is None."""
    if pricing is None:
        return 0.0
    return (
        (input_tokens / 1000) * pricing.input_per_1k
        + (output_tokens / 1000) * pricing.output_per_1k
    )


# ---------------------------------------------------------------------------
# measure_latency
# ---------------------------------------------------------------------------

def measure_latency(
    prompt: str,
    *,
    stream: bool = False,
    generation_config: dict | None = None,
    pricing: Pricing | None = None,
    provider: str | None = None,
) -> LatencyResult:
    """Measure latency (and TTFT when *stream=True*) of an LLM call."""
    provider = _resolve_provider(provider)
    if provider == "gemini":
        return _measure_gemini(prompt, stream=stream,
                               generation_config=generation_config, pricing=pricing)
    if provider == "groq":
        return _measure_groq(prompt, stream=stream,
                             generation_config=generation_config, pricing=pricing)
    raise ValueError(f"Unsupported provider: {provider}")


# -- Gemini ----------------------------------------------------------------

def _measure_gemini(
    prompt: str,
    *,
    stream: bool,
    generation_config: dict | None,
    pricing: Pricing | None,
) -> LatencyResult:
    client, model = _get_gemini_client()
    config = dict(generation_config or {})

    if not stream:
        t0 = perf_counter()
        response = client.models.generate_content(
            model=model, contents=prompt, config=config,
        )
        total_s = perf_counter() - t0

        output_text = (getattr(response, "text", "") or "").strip()

        input_tokens, output_tokens = _gemini_tokens(response)
        if not input_tokens:
            input_tokens = count_tokens(prompt, provider="gemini")
        if not output_tokens:
            output_tokens = count_tokens(output_text, provider="gemini") if output_text else 0

        tps = output_tokens / total_s if total_s > 0 else None
        cost = estimate_cost(input_tokens, output_tokens, pricing)

        return LatencyResult(
            ttft_s=None, total_s=total_s, tps=tps,
            input_tokens=input_tokens, output_tokens=output_tokens,
            output_text=output_text,
            meta={"cost": cost, "provider": "gemini"},
        )

    # --- streaming ---
    t0 = perf_counter()
    t_first: float | None = None
    output_text = ""

    try:
        stream_iter = client.models.generate_content_stream(
            model=model, contents=prompt, config=config,
        )
    except AttributeError:
        raise RuntimeError(
            "Streaming no soportado por el SDK actual de Gemini."
        )

    usage_meta = None
    for chunk in stream_iter:
        chunk_text = getattr(chunk, "text", "") or ""
        if chunk_text and t_first is None:
            t_first = perf_counter()
        output_text += chunk_text
        um = getattr(chunk, "usage_metadata", None)
        if um:
            usage_meta = um

    t_end = perf_counter()
    total_s = t_end - t0
    ttft_s = (t_first - t0) if t_first is not None else None

    input_tokens = _safe_int(getattr(usage_meta, "prompt_token_count", None)) if usage_meta else None
    output_tokens = _safe_int(getattr(usage_meta, "candidates_token_count", None)) if usage_meta else None

    if not input_tokens:
        input_tokens = count_tokens(prompt, provider="gemini")
    if not output_tokens:
        output_text_stripped = output_text.strip()
        output_tokens = count_tokens(output_text_stripped, provider="gemini") if output_text_stripped else 0

    tps = None
    if t_first is not None and (t_end - t_first) > 0:
        tps = output_tokens / (t_end - t_first)

    cost = estimate_cost(input_tokens, output_tokens, pricing)
    return LatencyResult(
        ttft_s=ttft_s, total_s=total_s, tps=tps,
        input_tokens=input_tokens, output_tokens=output_tokens,
        output_text=output_text.strip(),
        meta={"cost": cost, "provider": "gemini"},
    )


def _gemini_tokens(response: Any) -> tuple[int | None, int | None]:
    """Extract token counts from a Gemini response."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    return (
        _safe_int(getattr(usage, "prompt_token_count", None)),
        _safe_int(getattr(usage, "candidates_token_count", None)),
    )


# -- Groq -----------------------------------------------------------------

def _measure_groq(
    prompt: str,
    *,
    stream: bool,
    generation_config: dict | None,
    pricing: Pricing | None,
) -> LatencyResult:
    client, model = _get_groq_client()
    gen = dict(generation_config or {})

    # Map Gemini-style keys to Groq/OpenAI keys
    temperature = gen.pop("temperature", None)
    max_tokens = gen.pop("max_output_tokens", gen.pop("max_tokens", 1024))

    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature

    if not stream:
        t0 = perf_counter()
        response = client.chat.completions.create(**kwargs)
        total_s = perf_counter() - t0

        output_text = (response.choices[0].message.content or "").strip()
        input_tokens = getattr(response.usage, "prompt_tokens", None) if response.usage else None
        output_tokens = getattr(response.usage, "completion_tokens", None) if response.usage else None

        if not input_tokens:
            input_tokens = count_tokens(prompt, provider="groq")
        if not output_tokens:
            output_tokens = count_tokens(output_text, provider="groq") if output_text else 0

        tps = output_tokens / total_s if total_s > 0 else None
        cost = estimate_cost(input_tokens, output_tokens, pricing)

        return LatencyResult(
            ttft_s=None, total_s=total_s, tps=tps,
            input_tokens=input_tokens, output_tokens=output_tokens,
            output_text=output_text,
            meta={"cost": cost, "provider": "groq"},
        )

    # --- streaming ---
    kwargs["stream"] = True
    t0 = perf_counter()
    t_first: float | None = None
    output_text = ""
    usage_data = None

    stream_resp = client.chat.completions.create(**kwargs)
    for chunk in stream_resp:
        if not chunk.choices:
            # Final chunk may carry usage only
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage
            continue
        delta = chunk.choices[0].delta
        chunk_text = getattr(delta, "content", "") or ""
        if chunk_text and t_first is None:
            t_first = perf_counter()
        output_text += chunk_text
        if hasattr(chunk, "usage") and chunk.usage:
            usage_data = chunk.usage

    t_end = perf_counter()
    total_s = t_end - t0
    ttft_s = (t_first - t0) if t_first is not None else None

    input_tokens = getattr(usage_data, "prompt_tokens", None) if usage_data else None
    output_tokens = getattr(usage_data, "completion_tokens", None) if usage_data else None

    if not input_tokens:
        input_tokens = count_tokens(prompt, provider="groq")
    if not output_tokens:
        stripped = output_text.strip()
        output_tokens = count_tokens(stripped, provider="groq") if stripped else 0

    tps = None
    if t_first is not None and (t_end - t_first) > 0:
        tps = output_tokens / (t_end - t_first)

    cost = estimate_cost(input_tokens, output_tokens, pricing)
    return LatencyResult(
        ttft_s=ttft_s, total_s=total_s, tps=tps,
        input_tokens=input_tokens, output_tokens=output_tokens,
        output_text=output_text.strip(),
        meta={"cost": cost, "provider": "groq"},
    )


# ---------------------------------------------------------------------------
# BudgetChecker
# ---------------------------------------------------------------------------

class BudgetChecker:
    """Validates that a request stays within a cost budget."""

    def __init__(
        self,
        max_cost_usd: float,
        pricing: Pricing,
        *,
        strict: bool = True,
    ) -> None:
        self.max_cost_usd = max_cost_usd
        self.pricing = pricing
        self.strict = strict

    def check(self, input_tokens: int, output_tokens: int) -> dict:
        """Return status dict; raise *BudgetExceededError* in strict mode."""
        cost = estimate_cost(input_tokens, output_tokens, self.pricing)
        if cost > self.max_cost_usd:
            if self.strict:
                raise BudgetExceededError(
                    f"Presupuesto excedido: ${cost:.6f} > límite ${self.max_cost_usd:.6f}"
                )
            return {"ok": False, "cost": cost, "max": self.max_cost_usd}
        return {"ok": True, "cost": cost, "max": self.max_cost_usd}


# ---------------------------------------------------------------------------
# Streaming generators (for UI real-time display)
# ---------------------------------------------------------------------------

def stream_chunks(
    prompt: str,
    *,
    generation_config: dict | None = None,
    provider: str | None = None,
    _metrics_out: dict | None = None,
):
    """Yield text chunks from a streaming LLM call.

    If *_metrics_out* is a dict it will be filled **in-place** with timing
    and token information once the stream finishes.  This lets callers
    (e.g. Streamlit) display chunks in real-time while still capturing
    TTFT / total_s / tps afterwards.
    """
    provider = _resolve_provider(provider)
    if provider == "gemini":
        yield from _stream_chunks_gemini(prompt, generation_config, _metrics_out)
    elif provider == "groq":
        yield from _stream_chunks_groq(prompt, generation_config, _metrics_out)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _stream_chunks_gemini(prompt, generation_config, metrics_out):
    client, model = _get_gemini_client()
    config = dict(generation_config or {})

    t0 = perf_counter()
    t_first = None
    output_text = ""
    usage_meta = None

    stream_iter = client.models.generate_content_stream(
        model=model, contents=prompt, config=config,
    )
    for chunk in stream_iter:
        chunk_text = getattr(chunk, "text", "") or ""
        if chunk_text and t_first is None:
            t_first = perf_counter()
        if chunk_text:
            output_text += chunk_text
            yield chunk_text
        um = getattr(chunk, "usage_metadata", None)
        if um:
            usage_meta = um

    t_end = perf_counter()

    if metrics_out is not None:
        in_tok = _safe_int(getattr(usage_meta, "prompt_token_count", None)) if usage_meta else None
        out_tok = _safe_int(getattr(usage_meta, "candidates_token_count", None)) if usage_meta else None
        if not in_tok:
            in_tok = count_tokens(prompt, provider="gemini")
        if not out_tok:
            out_tok = count_tokens(output_text.strip(), provider="gemini") if output_text.strip() else 0
        tps = None
        if t_first is not None and (t_end - t_first) > 0:
            tps = out_tok / (t_end - t_first)
        metrics_out.update(
            ttft_s=(t_first - t0) if t_first else None,
            total_s=t_end - t0,
            tps=tps,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )


def _stream_chunks_groq(prompt, generation_config, metrics_out):
    client, model = _get_groq_client()
    gen = dict(generation_config or {})
    temperature = gen.pop("temperature", None)
    max_tokens = gen.pop("max_output_tokens", gen.pop("max_tokens", 1024))

    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "stream": True,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    t0 = perf_counter()
    t_first = None
    output_text = ""
    usage_data = None

    stream_resp = client.chat.completions.create(**kwargs)
    for chunk in stream_resp:
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage
            continue
        delta = chunk.choices[0].delta
        chunk_text = getattr(delta, "content", "") or ""
        if chunk_text and t_first is None:
            t_first = perf_counter()
        if chunk_text:
            output_text += chunk_text
            yield chunk_text
        if hasattr(chunk, "usage") and chunk.usage:
            usage_data = chunk.usage

    t_end = perf_counter()

    if metrics_out is not None:
        in_tok = getattr(usage_data, "prompt_tokens", None) if usage_data else None
        out_tok = getattr(usage_data, "completion_tokens", None) if usage_data else None
        if not in_tok:
            in_tok = count_tokens(prompt, provider="groq")
        if not out_tok:
            out_tok = count_tokens(output_text.strip(), provider="groq") if output_text.strip() else 0
        tps = None
        if t_first is not None and (t_end - t_first) > 0:
            tps = out_tok / (t_end - t_first)
        metrics_out.update(
            ttft_s=(t_first - t0) if t_first else None,
            total_s=t_end - t0,
            tps=tps,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
