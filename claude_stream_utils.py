"""
claude_stream_utils.py
----------------------
Utility for robust Claude API streaming with automatic retry on idle
timeout and other transient errors.

Error handled:
    APIStatusError / APIConnectionError:
        "Stream idle timeout - partial response received"

Usage:
    from claude_stream_utils import stream_with_retry, complete_with_retry

    # Streaming (yields text chunks)
    for chunk in stream_with_retry(messages=[{"role": "user", "content": "Hello"}]):
        print(chunk, end="", flush=True)

    # Non-streaming (returns full text)
    text = complete_with_retry(messages=[{"role": "user", "content": "Hello"}])
"""

import time
import anthropic

# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_MODEL         = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS    = 8096
DEFAULT_MAX_RETRIES   = 4          # total attempts = 1 + retries
DEFAULT_BASE_DELAY    = 2.0        # seconds — first retry wait
DEFAULT_MAX_DELAY     = 30.0       # seconds — cap on exponential back-off
DEFAULT_TIMEOUT       = 600.0      # seconds — httpx read timeout per request

# Errors whose message text signals a retriable stream problem
_RETRIABLE_PHRASES = (
    "stream idle timeout",
    "partial response received",
    "connection error",
    "connection reset",
    "read timeout",
    "timeout",
    "overloaded",
    "529",
    "500",
    "502",
    "503",
)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _is_retriable(exc: Exception) -> bool:
    """Return True when *exc* is a transient error worth retrying."""
    msg = str(exc).lower()
    if isinstance(exc, (anthropic.APIConnectionError,
                        anthropic.APITimeoutError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        # 429 (rate-limit), 500, 502, 503, 529 (overloaded)
        if exc.status_code in (429, 500, 502, 503, 529):
            return True
    # Catch stream-idle / partial-response by message text
    return any(phrase in msg for phrase in _RETRIABLE_PHRASES)


def _backoff(attempt: int, base: float = DEFAULT_BASE_DELAY,
             cap: float = DEFAULT_MAX_DELAY) -> float:
    """Exponential back-off: base * 2^attempt, capped at *cap* seconds."""
    return min(base * (2 ** attempt), cap)


def _make_client(timeout: float = DEFAULT_TIMEOUT) -> anthropic.Anthropic:
    """Create an Anthropic client with an explicit read timeout."""
    return anthropic.Anthropic(
        timeout=anthropic.Timeout(
            connect=10.0,
            read=timeout,
            write=30.0,
            pool=10.0,
        )
    )


# ── Public API ───────────────────────────────────────────────────────────────

def stream_with_retry(
    messages: list,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
    **extra_kwargs,
):
    """
    Stream a Claude response, retrying on idle-timeout or transient errors.

    Yields
    ------
    str
        Successive text delta chunks from the model.

    Parameters
    ----------
    messages    : list of {"role": ..., "content": ...} dicts
    model       : Claude model ID
    max_tokens  : upper bound on response tokens
    system      : optional system prompt string
    max_retries : how many times to retry after the first failure
    base_delay  : initial back-off wait in seconds
    max_delay   : maximum back-off wait in seconds
    timeout     : per-request read timeout in seconds
    **extra_kwargs : forwarded to client.messages.stream()
    """
    client = _make_client(timeout)
    kwargs = dict(model=model, max_tokens=max_tokens,
                  messages=messages, **extra_kwargs)
    if system:
        kwargs["system"] = system

    last_exc = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            wait = _backoff(attempt - 1, base_delay, max_delay)
            print(f"\n[claude_stream_utils] Retry {attempt}/{max_retries} "
                  f"after {wait:.1f}s  (reason: {last_exc})")
            time.sleep(wait)

        try:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
            return  # success — stop retrying

        except Exception as exc:
            last_exc = exc
            if not _is_retriable(exc):
                raise
            if attempt == max_retries:
                print(f"\n[claude_stream_utils] All {max_retries} retries "
                      f"exhausted. Raising last error.")
                raise


def complete_with_retry(
    messages: list,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
    **extra_kwargs,
) -> str:
    """
    Non-streaming Claude call with retry on idle-timeout / transient errors.

    Returns
    -------
    str
        The full response text.

    Parameters
    ----------
    Same as stream_with_retry.
    """
    client = _make_client(timeout)
    kwargs = dict(model=model, max_tokens=max_tokens,
                  messages=messages, **extra_kwargs)
    if system:
        kwargs["system"] = system

    last_exc = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            wait = _backoff(attempt - 1, base_delay, max_delay)
            print(f"\n[claude_stream_utils] Retry {attempt}/{max_retries} "
                  f"after {wait:.1f}s  (reason: {last_exc})")
            time.sleep(wait)

        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text

        except Exception as exc:
            last_exc = exc
            if not _is_retriable(exc):
                raise
            if attempt == max_retries:
                print(f"\n[claude_stream_utils] All {max_retries} retries "
                      f"exhausted. Raising last error.")
                raise

    return ""  # unreachable, satisfies type-checkers


# ── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Streaming response (with auto-retry):\n")
    for chunk in stream_with_retry(
        messages=[{"role": "user", "content": "Say hello in one sentence."}]
    ):
        print(chunk, end="", flush=True)
    print("\n\nDone.")
