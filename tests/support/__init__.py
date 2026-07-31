"""Shared test helpers: settings/fixtures constants, mock upstreams, audit polling.

Kept out of `conftest.py` so tests can import these directly (a plain function is
easier to reason about than a fixture when only one test needs it).
"""

from tests.support.audit import make_audit, wait_for_audit_row
from tests.support.fixtures import (
    FAKE_EMAIL,
    FAKE_OPENAI_KEY,
    HARD_INJECTION,
    SECRET_PROMPT,
    SOFT_INJECTION,
    make_settings,
)
from tests.support.upstream import sse_handler

__all__ = [
    "FAKE_EMAIL",
    "FAKE_OPENAI_KEY",
    "HARD_INJECTION",
    "SECRET_PROMPT",
    "SOFT_INJECTION",
    "make_audit",
    "make_settings",
    "sse_handler",
    "wait_for_audit_row",
]
