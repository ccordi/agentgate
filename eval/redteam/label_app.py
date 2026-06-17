"""Streamlit gold-set labeler — click Benign / Malicious / Unknown, then Save.

Run from the repo root:

    uv run --extra label streamlit run eval/redteam/label_app.py

Reads an unlabeled JSONL (default `gold_set_unlabeled.jsonl`) and writes a labeled JSONL
(default `gold_set_labeled.jsonl`). Labeling is **blind by design** — only the text is
shown, never the corpus category or the scanner's score, so the human judgment is
independent (that independence is what makes the judge-vs-human κ meaningful).

  Benign → label 0    Malicious → label 1    Unknown → label null (excluded from κ)

Output uses the same schema as the corpus, so `python -m eval.redteam report` picks it up
automatically (see loader.resolve_gold_path).
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Make the `eval` package importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from eval.redteam.loader import load_jsonl, write_jsonl  # noqa: E402
from eval.redteam.schema import CorpusItem, LabelOrigin  # noqa: E402

DEFAULT_IN = "eval/redteam/corpus/gold_set_unlabeled.jsonl"
DEFAULT_OUT = "eval/redteam/corpus/gold_set_labeled.jsonl"

CHOICES = ["Unknown", "Benign", "Malicious"]
CHOICE_TO_LABEL = {"Unknown": None, "Benign": 0, "Malicious": 1}
LABEL_TO_CHOICE = {None: "Unknown", 0: "Benign", 1: "Malicious"}


def _key(item_id: str) -> str:
    return f"choice_{item_id}"


def load_items(in_path: str, out_path: str) -> None:
    """Load input items; preload any choices already saved in the output file (resume)."""
    items = list(load_jsonl(Path(in_path)))
    prior: dict[str, int | None] = {}
    op = Path(out_path)
    if op.exists():
        for it in load_jsonl(op):
            prior[it.id] = it.label
    for it in items:
        # Seed widget state before the radios are created (Streamlit's supported pattern).
        st.session_state[_key(it.id)] = LABEL_TO_CHOICE.get(prior.get(it.id), "Unknown")
    # NB: key must not be "items" — that collides with the session-state mapping's
    # .items() method, so attribute access would return the method, not our list.
    st.session_state.gold_items = items


def save_items(out_path: str) -> tuple[int, int]:
    """Write current choices to the output JSONL. Returns (total, binary-labeled)."""
    out: list[CorpusItem] = []
    for it in st.session_state.gold_items:
        choice = st.session_state.get(_key(it.id), "Unknown")
        out.append(CorpusItem(
            id=it.id, source=it.source, text=it.text,
            label=CHOICE_TO_LABEL[choice],
            label_origin=LabelOrigin.HUMAN, category=None, meta=it.meta,
        ))
    n = write_jsonl(Path(out_path), out)
    binary = sum(1 for o in out if o.label is not None)
    return n, binary


st.set_page_config(page_title="Red-team gold-set labeler", layout="centered")
st.title("Red-team gold-set labeler")

# Streamlit discards the state of widgets that aren't rendered on a given run. With
# "show only unlabeled" on, labeling an item filters its radio out of the next render,
# which would drop its choice from session_state (and reset the count). Re-asserting each
# known choice key as plain session state every run keeps hidden items' labels alive.
# Must run BEFORE the radios are created.
if "gold_items" in st.session_state:
    for _it in st.session_state.gold_items:
        _k = _key(_it.id)
        if _k in st.session_state:
            st.session_state[_k] = st.session_state[_k]

with st.sidebar:
    st.header("Files")
    in_path = st.text_input("Input JSONL", DEFAULT_IN)
    out_path = st.text_input("Output JSONL", DEFAULT_OUT)
    if st.button("Load / reload", use_container_width=True):
        try:
            load_items(in_path, out_path)
            st.success(f"Loaded {len(st.session_state.gold_items)} items")
        except FileNotFoundError:
            st.error(f"Not found: {in_path}")
    # Save lives in the sidebar (which stays put while you scroll the items).
    if st.button("💾 Save", type="primary", use_container_width=True,
                 disabled="gold_items" not in st.session_state):
        total, binary = save_items(out_path)
        st.success(f"Saved {total} ({binary} labeled 0/1, {total - binary} Unknown)")
    st.caption("Label from the text alone — category & scanner score are intentionally hidden.")

if "gold_items" not in st.session_state:
    st.info("Set the input/output paths in the sidebar and click **Load / reload**.")
    st.stop()

items = st.session_state.gold_items
done = sum(1 for it in items if st.session_state.get(_key(it.id), "Unknown") != "Unknown")
st.progress(done / len(items) if items else 0.0, text=f"{done}/{len(items)} labeled (non-Unknown)")

only_unlabeled = st.checkbox("Show only unlabeled", value=False)

st.divider()

for idx, it in enumerate(items, 1):
    if only_unlabeled and st.session_state.get(_key(it.id), "Unknown") != "Unknown":
        continue
    st.markdown(f"**{idx}.** `{it.source}`")
    # Wrapping, auto-height box. html.escape() neutralizes any markup in the payload so it
    # can't inject HTML; pre-wrap keeps newlines, overflow-wrap breaks long unbroken tokens.
    st.markdown(
        "<div style='white-space:pre-wrap; overflow-wrap:anywhere; font-family:monospace; "
        "font-size:0.9rem; color:#24292f; background:#f6f8fa; border:1px solid #d0d7de; "
        f"border-radius:6px; padding:8px 12px; margin-bottom:6px;'>{html.escape(it.text)}</div>",
        unsafe_allow_html=True,
    )
    st.radio("label", CHOICES, key=_key(it.id), horizontal=True, label_visibility="collapsed")
    st.divider()

if st.button("💾 Save", key="save_bottom", type="primary", use_container_width=True):
    total, binary = save_items(out_path)
    st.success(f"Saved {total} items ({binary} labeled 0/1, {total - binary} Unknown) → {out_path}")
