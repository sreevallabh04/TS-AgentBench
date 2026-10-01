"""LLM client tests: caching, budgets, quota classification and key rotation.

None of these make a network call. The provider call is stubbed, because what needs
testing is the surrounding machinery -- whether a cached response is replayed, whether a
budget ceiling actually stops spending, and whether a daily quota error rotates to a
different key instead of being retried forever. Those are the parts that decide whether
an overnight experiment finishes or burns its budget on calls that cannot succeed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.llm import (  # noqa: E402
    CacheMiss, LLMBudgetExceeded, LLMClient, LLMResponse, _is_daily_quota,
    _is_transient,
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for name in list(dict(**{k: v for k, v in __import__("os").environ.items()})):
        if name.startswith(("GEMINI_API_KEY", "GROQ_API_KEY")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "key-one")
    return LLMClient(cache_dir=tmp_path, requests_per_minute=0, max_calls=10)


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #


def test_daily_quota_is_distinguished_from_per_minute_quota():
    """The whole point of the distinction: one clears by waiting, the other never does."""
    daily = ("{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', "
             "'quotaValue': '500'}")
    per_minute = ("{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', "
                  "'quotaValue': '15'}")
    assert _is_daily_quota(daily)
    assert not _is_daily_quota(per_minute)


def test_transient_errors_are_recognised():
    assert _is_transient(Exception("429 RESOURCE_EXHAUSTED"))
    assert _is_transient(Exception("503 Service Unavailable"))
    assert _is_transient(Exception("connection reset"))
    assert not _is_transient(Exception("400 INVALID_ARGUMENT: bad request"))


# --------------------------------------------------------------------------- #
# Caching and budget
# --------------------------------------------------------------------------- #


def test_response_is_cached_and_replayed_without_a_second_call(client, monkeypatch):
    calls = []

    def fake(prompt, system):
        calls.append(prompt)
        return LLMResponse(text='{"action":"accept"}', model=client.model)

    monkeypatch.setattr(client, "_call_provider", fake)
    first = client.generate("prompt A")
    second = client.generate("prompt A")

    assert len(calls) == 1, "the second call should have been served from cache"
    assert first.text == second.text
    assert second.cached is True
    assert client.n_calls == 1 and client.n_cache_hits == 1


def test_cache_key_separates_providers_and_models(client, monkeypatch):
    """A Groq answer must never be replayed as a Gemini answer.

    The two families are reported as independent replications; silently sharing a cache
    entry between them would collapse that into one.
    """
    monkeypatch.setattr(client, "_call_provider",
                        lambda p, s: LLMResponse(text="x", model=client.model))
    client.generate("same prompt")

    other = LLMClient(provider="groq", model="openai/gpt-oss-120b",
                      cache_dir=client.cache_dir, requests_per_minute=0)
    assert other._cache_key("same prompt", None) != client._cache_key("same prompt", None)


def test_budget_ceiling_stops_spending(client, monkeypatch):
    monkeypatch.setattr(client, "_call_provider",
                        lambda p, s: LLMResponse(text="x", model=client.model))
    client.max_calls = 3
    for i in range(3):
        client.generate(f"prompt {i}")
    with pytest.raises(LLMBudgetExceeded):
        client.generate("one too many")
    assert client.n_calls == 3


def test_cache_only_refuses_to_call_the_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-one")
    c = LLMClient(cache_dir=tmp_path, cache_only=True, requests_per_minute=0)
    with pytest.raises(CacheMiss):
        c.generate("never seen before")


def test_failed_responses_are_not_cached(client, monkeypatch):
    """A provider failure must not become a permanent cached 'answer'.

    Caching an error would make a quota outage look like a stable model decision on
    every subsequent run, which is exactly the silent corruption the arms guard against.
    """
    monkeypatch.setattr(
        client, "_call_provider",
        lambda p, s: LLMResponse(text="", model=client.model, error="429 quota"),
    )
    client.generate("prompt")
    assert client.n_errors == 1
    assert client._read_cache(client._cache_key("prompt", None)) is None


# --------------------------------------------------------------------------- #
# Key rotation
# --------------------------------------------------------------------------- #


def test_keys_are_discovered_from_numbered_environment_variables(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-two")
    monkeypatch.setenv("GEMINI_API_KEY_3", "key-three")
    assert client._api_keys() == ["key-one", "key-two", "key-three"]


def test_duplicate_keys_are_collapsed(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-one")
    assert client._api_keys() == ["key-one"]


def test_groq_reads_its_own_key_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    c = LLMClient(provider="groq", cache_dir=tmp_path, requests_per_minute=0)
    assert c._api_keys() == ["groq-key"]


def test_daily_quota_rotates_to_the_next_key(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-two")
    assert client._current_key() == "key-one"
    rotated = client._handle_quota(
        Exception("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
        "key-one",
    )
    assert rotated is True
    assert client._current_key() == "key-two"


def test_rotation_stops_when_every_key_is_exhausted(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-two")
    daily = Exception("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert client._handle_quota(daily, "key-one") is True
    assert client._handle_quota(daily, "key-two") is False, (
        "with no keys left the client must give up rather than loop"
    )


def test_per_minute_quota_does_not_burn_a_key(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-two")
    rotated = client._handle_quota(
        Exception("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"), "key-one"
    )
    assert rotated is False
    assert client._current_key() == "key-one"


def test_missing_key_raises_a_useful_error(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        monkeypatch.delenv(name, raising=False)
    c = LLMClient(cache_dir=tmp_path, requests_per_minute=0)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        c._current_key()


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


def test_rate_limiter_admits_up_to_its_budget_without_blocking():
    from agents.llm import _RateLimiter

    limiter = _RateLimiter(per_minute=5)
    import time

    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    assert time.monotonic() - start < 1.0, "the first five should not have waited"


def test_rate_limiter_disabled_at_zero():
    from agents.llm import _RateLimiter
    import time

    limiter = _RateLimiter(per_minute=0)
    start = time.monotonic()
    for _ in range(100):
        limiter.acquire()
    assert time.monotonic() - start < 1.0


def test_reasoning_effort_separates_cache_entries(tmp_path):
    """Two efforts are two experiments; one must not read the other's answers.

    Effort was originally absent from the key, so a medium-effort run could have been
    served a low-effort reply with nothing in the output to show it. The field is added
    to the key only when set, which is what keeps the thousands of responses cached
    before reasoning models existed readable at their original keys.
    """
    def client(effort):
        return LLMClient(provider="groq", model="openai/gpt-oss-20b",
                         cache_dir=tmp_path, requests_per_minute=0,
                         reasoning_effort=effort)

    none_key = client(None)._cache_key("p", "s")
    low_key = client("low")._cache_key("p", "s")
    medium_key = client("medium")._cache_key("p", "s")
    assert len({none_key, low_key, medium_key}) == 3

    # Unset effort must reproduce the key the non-reasoning runs were cached under:
    # the field is omitted from the payload entirely, not serialised as null.
    import hashlib, json
    legacy = hashlib.sha256(json.dumps({
        "provider": "groq", "model": "openai/gpt-oss-20b", "temperature": 0.0,
        "max_output_tokens": 1024, "system": "s", "prompt": "p",
    }, sort_keys=True).encode("utf-8")).hexdigest()
    assert none_key == legacy
