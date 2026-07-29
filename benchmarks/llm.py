"""
Resilient LLM client for the reader and judge stages.

Why this exists rather than reusing src/llm_extractor.py: that module's Groq path sets
`max_retries = len(clients)`, so with a single API key it makes exactly ONE attempt with
no backoff (src/llm_extractor.py:304). Any transient 429 raises. Over a run with tens of
thousands of calls that is a guaranteed abort. This client does bounded exponential
backoff with jitter and honours Retry-After.

Providers: groq (default), openai, anthropic. Keys come from the environment.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

_RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "too many requests",
                       "overloaded", "capacity", "503", "502", "504", "timeout")


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    failures: int = 0

    def merge(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.retries += other.retries
        self.failures += other.failures

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "failures": self.failures,
        }


def _retry_after_seconds(message: str) -> Optional[float]:
    """Groq and OpenAI both embed a wait hint in the error body; use it when present."""
    for pat in (r"try again in ([0-9.]+)s", r"retry[- ]after[\"':\s]+([0-9.]+)"):
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


class LLMClient:
    """
    Thread-safe chat wrapper with retry/backoff and usage accounting.

    A single instance is shared across the reader and judge inside one worker process.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_attempts: int = 8,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 120.0,
    ):
        self.provider = (provider or os.getenv("BENCH_LLM_PROVIDER", "groq")).lower()
        self.model = model or os.getenv("BENCH_LLM_MODEL", "llama-3.3-70b-versatile")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout

        self.usage = Usage()
        self._lock = threading.Lock()
        self._clients: List[object] = []
        self._idx = 0
        self._build_clients()

    # ---------------------------------------------------------------- clients

    def _keys(self, *names: str) -> List[str]:
        keys: List[str] = []
        for name in names:
            v = os.getenv(name)
            if v and v.strip() and not v.strip().startswith(("gsk-your", "sk-your", "sk-ant-your")):
                keys.append(v.strip())
        # de-dup, preserve order
        return list(dict.fromkeys(keys))

    def _build_clients(self) -> None:
        if self.provider == "groq":
            from groq import Groq
            keys = self._keys("GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2",
                              "GROQ_API_KEY_3", "BENCH_GROQ_API_KEY")
            if not keys:
                raise RuntimeError("No Groq API key found (set GROQ_API_KEY)")
            # max_retries=0: this class owns retry policy, not the SDK.
            self._clients = [Groq(api_key=k, max_retries=0, timeout=self.timeout) for k in keys]

        elif self.provider == "openai":
            from openai import OpenAI
            keys = self._keys("OPENAI_API_KEY", "BENCH_OPENAI_API_KEY")
            if not keys:
                raise RuntimeError("No OpenAI API key found (set OPENAI_API_KEY)")
            self._clients = [OpenAI(api_key=k, max_retries=0, timeout=self.timeout) for k in keys]

        elif self.provider == "anthropic":
            from anthropic import Anthropic
            keys = self._keys("ANTHROPIC_API_KEY", "BENCH_ANTHROPIC_API_KEY")
            if not keys:
                raise RuntimeError("No Anthropic API key found (set ANTHROPIC_API_KEY)")
            self._clients = [Anthropic(api_key=k, max_retries=0, timeout=self.timeout)
                             for k in keys]
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _next_client(self) -> object:
        with self._lock:
            client = self._clients[self._idx]
            self._idx = (self._idx + 1) % len(self._clients)
            return client

    # ------------------------------------------------------------------ call

    def _invoke(self, client: object, system: str, user: str,
                max_tokens: int, temperature: float) -> tuple[str, int, int]:
        if self.provider == "anthropic":
            resp = client.messages.create(   # type: ignore[attr-defined]
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return text, resp.usage.input_tokens, resp.usage.output_tokens

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = client.chat.completions.create(   # type: ignore[attr-defined]
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        return text, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0

    def chat(
        self,
        user: str,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """
        Returns the response text, or None if every attempt failed.

        None is deliberate: one unanswerable question should degrade that question's
        score, not abort a multi-hour benchmark. Failures are counted in `usage` so the
        report can state how many questions were lost.
        """
        last_error: Optional[str] = None

        for attempt in range(self.max_attempts):
            client = self._next_client()
            try:
                text, tin, tout = self._invoke(client, system, user, max_tokens, temperature)
                with self._lock:
                    self.usage.calls += 1
                    self.usage.input_tokens += tin
                    self.usage.output_tokens += tout
                return text
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise varied types
                last_error = str(exc)
                transient = any(m in last_error.lower() for m in _RATE_LIMIT_MARKERS)
                with self._lock:
                    self.usage.retries += 1
                if not transient or attempt == self.max_attempts - 1:
                    break
                hinted = _retry_after_seconds(last_error)
                delay = hinted if hinted is not None else min(
                    self.base_delay * (2 ** attempt), self.max_delay
                )
                time.sleep(delay + random.uniform(0, 0.5 * delay + 0.1))

        with self._lock:
            self.usage.failures += 1
        return None
