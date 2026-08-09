"""Secret handling. This repo publishes JSON to a public URL, so a leaked
key here is permanent — git history keeps it after the file is deleted."""
import os

import pytest

from mi.redact import REDACTED, assert_clean, clean


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("FMP_KEY", "abcd1234efgh5678")
    return "abcd1234efgh5678"


def test_removes_known_key_from_error_string(key):
    msg = f"HTTPError: https://fmp.com/v3/quote?apikey={key}&symbol=MU"
    assert key not in clean(msg)
    assert REDACTED in clean(msg)


def test_removes_unknown_credential_params():
    """Catches keys for providers added later, before they leak once."""
    out = clean("GET https://api.example.com/x?access_token=zzzzzzzzzzzzzz")
    assert "zzzzzzzzzzzzzz" not in out


def test_recurses_through_payload(key):
    payload = {"a": [{"b": f"token={key}"}], "c": ("x", f"apikey={key}")}
    assert key not in repr(clean(payload))


def test_guard_raises_on_survivor(key):
    with pytest.raises(RuntimeError, match="ABORTING"):
        assert_clean({"leaked": key})


def test_guard_passes_clean_payload(key):
    assert_clean({"ok": "no secrets here"})


def test_non_strings_pass_through():
    assert clean(3.14) == 3.14
    assert clean(None) is None
