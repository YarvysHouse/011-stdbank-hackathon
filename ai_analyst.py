""" The Gemini-backed analyst behind page 7.

The model never sees the data and never runs code against it. It gets a fixed set
of tools - the same aggregations the tabs already render - and calls them by name
with validated arguments. Every number it quotes therefore comes out of the same
functions the dashboard was checked against, so a wrong answer is a wrong sentence
about right figures rather than an invented figure.

Deliberately not a pandas agent: `df.query(model_supplied_string)` would let a
prompt reach any column, any row, and any Python builtin `eval` exposes. The
artifacts are only 71 KB, but the reported financials in them are hand-keyed from
published PDFs and the wallet-share figures are commercially sensitive - the
narrow tool surface is what keeps the blast radius to "answered a question badly".

Key handling - the app reads, in order:

    1. st.secrets["GEMINI_API_KEY"]     Streamlit Cloud, or local .streamlit/secrets.toml
    2. os.environ["GEMINI_API_KEY"]     shell export, or a .env someone sourced

Validate the wiring without launching the dashboard::

    uv run python ai_analyst.py --check          # key resolves, and a live round-trip
    uv run python ai_analyst.py --models         # what this key can actually call
    uv run python ai_analyst.py --ask "..."      # one question through the full tool loop

The model is not pinned. Google retires names on its own schedule - a hardcoded
`gemini-2.5-flash` began 404ing in production with nothing here changed - so
`resolve_model()` picks the best model the key can see, and a 404 mid-session
re-discovers rather than failing. Set GEMINI_MODEL to override.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from sizing import ID_COLS, load_tables, missed_wallet, project, reliable_lines

# Only used when the key cannot list models at all - the real choice is made by
# `resolve_model()` against what the account can actually see. Do not pin here.
FALLBACK_MODEL = "gemini-flash-latest"
MAX_TOOL_TURNS = 8  # a question needing more than this is one the tools do not answer

SYSTEM = """You are a corporate banking analyst for Syn Bank, working inside a wallet-share
dashboard. Twenty listed South African entities are banked; the analysis measures what share of
each client's published financials the bank actually handles.

Rules:
- Every figure you state must come from a tool call. Never estimate, never carry a number over
  from memory, never do arithmetic the tools can do for you.
- If the tools cannot answer, say so plainly and name what is missing. Do not improvise.
- Wallet share is genuinely tiny - around 0.2% of reported financials. That is the finding, not
  an error. Do not apologise for it or assume a unit mistake.
- Amounts are ZAR. Format them the way the dashboard does: R 57.85tn, R 88.0bn, R 20.4m.
- Comparison rows where the computed figure exceeds 50% of reported are PDF extraction scale
  errors, and the tools already exclude them. Mention this only if asked about data quality.
- Answer like a banker briefing a relationship team: the number, what it means, what to do.
  Two or three short paragraphs at most. No bullet-point walls, no restating the question.
"""

# The tool surface. Declared to Gemini as function declarations; dispatched below.
# Kept deliberately parallel to what the tabs compute - nothing here can reach a
# row the dashboard does not already display.
TOOLS = [
    {
        "name": "portfolio_overview",
        "description": ("Book-wide position: entity and sector counts, transactions, total flow, "
                        "reported financials, wallet share carried, addressable gap, and modelled "
                        "fee revenue. Start here for any 'how big is the opportunity' question."),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "entity_detail",
        "description": ("Everything known about one client: flow, transaction count, wallet share, "
                        "missed wallet, its top reference types, growth rate and projected uplift. "
                        "Use for any question naming a specific company."),
        "parameters": {
            "type": "object",
            "properties": {"entity_name": {"type": "string",
                                           "description": "Company name, e.g. 'Glencore'. Partial match is fine."}},
            "required": ["entity_name"],
        },
    },
    {
        "name": "sector_rollup",
        "description": ("Every sector with its entity count, flow, reported financials, wallet "
                        "share and gap. Use to compare sectors or find where the gap concentrates."),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "opportunity_ranking",
        "description": ("Clients ranked by opportunity, with missed wallet, implied transaction "
                        "volume and modelled fee revenue. Use for 'who should we target' questions."),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "How many clients to return. Default 10."},
                "rank_by": {"type": "string", "enum": ["missed_amount", "implied_txns", "fee_revenue"],
                            "description": "Ranking measure. Default missed_amount."},
                "fee_bps": {"type": "integer", "description": "Fee on value routed, basis points. Default 15."},
                "fee_per_txn": {"type": "integer", "description": "Fee per transaction in rand. Default 5."},
            },
        },
    },
    {
        "name": "projection",
        "description": ("Published revenue compounded forward at each client's CAGR with bank share "
                        "held flat - the fee income arriving without winning new mandates. Use for "
                        "any question about the future, growth, or which client to defend."),
        "parameters": {
            "type": "object",
            "properties": {
                "horizon_years": {"type": "integer", "description": "Years to compound. Default 5."},
                "fee_bps": {"type": "integer", "description": "Fee on value routed, basis points. Default 15."},
                "limit": {"type": "integer", "description": "How many clients, by uplift. Default 10."},
            },
        },
    },
    {
        "name": "cross_border",
        "description": ("Counterparty countries with value routed and transaction counts, income "
                        "against payment. Use for trade corridor and FX questions."),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "Optional - restrict to one client."},
                "limit": {"type": "integer", "description": "How many countries. Default 10."},
            },
        },
    },
]


# --------------------------------------------------------------------------
# Dispatch - each returns plain JSON-able dicts, never a DataFrame
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _tables():
    """The frames, loaded once per process.

    Loaded here rather than imported from `entity_app` so the CLI checks do not
    have to execute a Streamlit script; both sides call the same `sizing` module.
    """
    summary, reference, counts, comparison, geo, projection_df = load_tables()
    return {
        "summary": summary, "counts": counts, "comparison": comparison,
        "geo": geo, "projection": projection_df,
        "ref_types": [c for c in reference.columns if c not in ID_COLS],
    }


def jsonable(obj):
    """Numpy scalars out, plain Python in.

    pandas hands back np.float64 and np.int64, which the SDK cannot serialise into
    a function response - and NaN is not valid JSON either, so it becomes null.
    """
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if pd.isna(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NaT or obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def _round(frame, digits=2):
    """Trim float noise so the model is not handed 14 significant figures to quote."""
    return frame.round(digits).replace({np.nan: None}).to_dict("records")


def _match(frame, name):
    hit = frame[frame["entity_name"].str.contains(str(name), case=False, na=False)]
    return hit


def portfolio_overview():
    t = _tables()
    book = missed_wallet(reliable_lines(t["comparison"]), t["summary"], 15, 5)
    proj = project(t["projection"], 5, 15)
    summary = t["summary"]

    reported = book["reported"].sum()
    return {
        "entities": int(len(summary)),
        "sectors": int(summary["sector"].nunique()),
        "transactions": int(summary["num_transactions"].sum()),
        "incomes_zar": float(summary["incomes"].sum()),
        "payments_zar": float(summary["payments"].sum()),
        "reported_financials_zar": float(reported),
        "carried_zar": float(book["computed"].sum()),
        "wallet_share_pct": float(book["computed"].sum() / reported * 100),
        "addressable_gap_zar": float(book["missed_amount"].sum()),
        "modelled_fee_revenue_zar": float(book["fee_revenue"].sum()),
        "fee_assumptions": "15 bps on value routed plus R5 per transaction",
        "projection_5y_uplift_zar": float(proj["bank_uplift"].sum()),
        "projection_note": f"{len(proj)} of {len(t['projection'])} clients carry a reported revenue line to compound",
    }


def entity_detail(entity_name: str):
    t = _tables()
    hit = _match(t["summary"], entity_name)
    if hit.empty:
        return {"error": f"No client matching '{entity_name}'.",
                "available": sorted(t["summary"]["entity_name"].tolist())}

    row = hit.iloc[0]
    eid = row["entity_id"]

    book = missed_wallet(reliable_lines(t["comparison"]), t["summary"], 15, 5)
    mine = book[book["entity_id"] == eid]
    proj = project(t["projection"], 5, 15)
    prow = proj[proj["entity_id"] == eid]

    counts = t["counts"]
    crow = counts[counts["entity_id"] == eid][t["ref_types"]].sum()
    top3 = crow.nlargest(3)

    out = {
        "entity_id": eid,
        "entity_name": row["entity_name"],
        "sector": row["sector"],
        "incomes_zar": float(row["incomes"]),
        "payments_zar": float(row["payments"]),
        "transactions": int(row["num_transactions"]),
        "top_reference_types": [{"reference_type": k, "transactions": int(v),
                                 "pct_of_client_transactions": round(v / crow.sum() * 100, 1)}
                                for k, v in top3.items()],
    }
    if mine.empty:
        out["comparison"] = "No reconciling reported line - this client cannot be sized against published financials."
    else:
        m = mine.iloc[0]
        out |= {
            "reported_financials_zar": float(m["reported"]),
            "carried_zar": float(m["computed"]),
            "wallet_share_pct": round(float(m["computed"] / m["reported"] * 100), 4),
            "missed_wallet_zar": float(m["missed_amount"]),
            "implied_transactions": None if pd.isna(m["implied_txns"]) else int(m["implied_txns"]),
            "modelled_fee_revenue_zar": float(m["fee_revenue"]),
        }
    if not prow.empty:
        p = prow.iloc[0]
        out |= {
            "revenue_cagr_pct": round(float(p["cagr_pct"]), 2),
            "base_year": str(p["base_year"]),  # 'FY2025', not an integer
            "base_revenue_zar": float(p["base_revenue"]),
            "projected_revenue_5y_zar": float(p["projected_revenue"]),
            "bank_fee_now_zar": float(p["bank_now"]),
            "bank_fee_5y_zar": float(p["bank_future"]),
            "uplift_5y_zar": float(p["bank_uplift"]),
        }
    return out


def sector_rollup():
    t = _tables()
    summary, book = t["summary"], missed_wallet(reliable_lines(t["comparison"]), t["summary"], 15, 5)

    flow = summary.groupby("sector", as_index=False).agg(
        entities=("entity_id", "nunique"), incomes=("incomes", "sum"),
        payments=("payments", "sum"), transactions=("num_transactions", "sum"))
    gap = book.groupby("sector", as_index=False).agg(
        reported=("reported", "sum"), carried=("computed", "sum"),
        missed=("missed_amount", "sum"), fee_revenue=("fee_revenue", "sum"))

    d = flow.merge(gap, on="sector", how="left")
    d["wallet_share_pct"] = d["carried"] / d["reported"] * 100
    return {"sectors": _round(d.sort_values("missed", ascending=False), 3)}


def opportunity_ranking(limit: int = 10, rank_by: str = "missed_amount",
                        fee_bps: int = 15, fee_per_txn: int = 5):
    t = _tables()
    book = missed_wallet(reliable_lines(t["comparison"]), t["summary"], fee_bps, fee_per_txn)
    if rank_by not in ("missed_amount", "implied_txns", "fee_revenue"):
        rank_by = "missed_amount"

    d = (book.sort_values(rank_by, ascending=False)
             .head(max(1, min(int(limit), 20)))[
                 ID_COLS + ["reported", "computed", "missed_amount", "avg_ticket",
                            "implied_txns", "fee_revenue"]])
    return {"ranked_by": rank_by,
            "fee_assumptions": f"{fee_bps} bps plus R{fee_per_txn} per transaction",
            "total_missed_zar": float(book["missed_amount"].sum()),
            "clients": _round(d, 2)}


def projection(horizon_years: int = 5, fee_bps: int = 15, limit: int = 10):
    t = _tables()
    horizon = max(1, min(int(horizon_years), 10))
    p = project(t["projection"], horizon, fee_bps)

    d = (p.sort_values("bank_uplift", ascending=False)
          .head(max(1, min(int(limit), 20)))[
              ID_COLS + ["cagr_pct", "base_year", "base_revenue", "projected_revenue",
                         "wallet_share_pct", "bank_now", "bank_future", "bank_uplift"]])
    return {
        "horizon_years": horizon,
        "fee_bps": fee_bps,
        "assumption": "bank share of each client held flat - this sizes growth already committed to, not share to be won",
        "total_projected_revenue_zar": float(p["projected_revenue"].sum()),
        "total_bank_fee_now_zar": float(p["bank_now"].sum()),
        "total_bank_fee_future_zar": float(p["bank_future"].sum()),
        "total_uplift_zar": float(p["bank_uplift"].sum()),
        "contracting_clients": p[p["cagr_pct"] < 0]["entity_name"].tolist(),
        "clients": _round(d, 3),
    }


def cross_border(entity_name: str | None = None, limit: int = 10):
    t = _tables()
    geo = t["geo"]
    if entity_name:
        geo = _match(geo, entity_name)
        if geo.empty:
            return {"error": f"No cross-border activity for '{entity_name}'."}

    wide = (geo.pivot_table(index="counterparty_country", columns="flow", values="value_zar",
                            aggfunc="sum", fill_value=0).reset_index())
    wide.columns.name = None
    for col in ("Income", "Payment"):
        if col not in wide:
            wide[col] = 0.0
    wide["total"] = wide["Income"] + wide["Payment"]

    counts = geo.groupby("counterparty_country")["txn_count"].sum()
    wide["transactions"] = wide["counterparty_country"].map(counts).astype(int)

    d = wide.sort_values("total", ascending=False).head(max(1, min(int(limit), 40)))
    return {
        "scope": entity_name or "all clients",
        "countries": int(geo["counterparty_country"].nunique()),
        "total_value_zar": float(geo["value_zar"].sum()),
        "total_income_zar": float(wide["Income"].sum()),
        "total_payment_zar": float(wide["Payment"].sum()),
        "stance": "net importer" if wide["Payment"].sum() > wide["Income"].sum() else "net exporter",
        "top_corridors": _round(d, 2),
    }


DISPATCH = {
    "portfolio_overview": portfolio_overview,
    "entity_detail": entity_detail,
    "sector_rollup": sector_rollup,
    "opportunity_ranking": opportunity_ranking,
    "projection": projection,
    "cross_border": cross_border,
}


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

def api_key() -> str | None:
    """st.secrets first, environment second. Never raises when Streamlit is absent."""
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip() or None
    except Exception:
        pass  # no Streamlit, or no secrets.toml - fall through to the environment
    return (os.environ.get("GEMINI_API_KEY") or "").strip() or None


def key_source() -> str:
    """Which of the two paths supplied the key - shown in the UI so wiring is debuggable."""
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return "st.secrets"
    except Exception:
        pass
    return "environment" if os.environ.get("GEMINI_API_KEY") else "not found"


def _client(key: str):
    from google import genai
    return genai.Client(api_key=key)


# --------------------------------------------------------------------------
# Model selection
#
# Google retires model names out from under deployed code - a pinned
# `gemini-2.5-flash` started returning 404 NOT_FOUND ("no longer available to
# new users") without anything here changing. So the model is discovered from
# what the key can actually see, rather than hardcoded and hoped for.
# --------------------------------------------------------------------------

# Families that cannot do what this panel needs, whatever they are called.
_EXCLUDE = ("embedding", "aqa", "imagen", "veo", "image", "tts", "audio",
            "live", "gemma", "learnlm", "vision")


def _model_rank(name: str) -> tuple:
    """Sort key for a candidate model. Higher is better.

    A `-latest` alias outranks everything numbered: pinning a version is what
    broke this panel in the first place, and Google keeps the alias current.
    Below that: newer generation, flash over pro (a high-frequency lookup panel
    wants latency and cost over depth), full over lite, stable over preview.
    """
    short = name.removeprefix("models/")

    generation = 0.0
    for part in short.replace("gemini-", "", 1).split("-"):
        try:
            generation = float(part)
            break
        except ValueError:
            continue

    return (99.0 if "latest" in short else generation,
            1 if "flash" in short else 0,
            0 if "lite" in short else 1,
            0 if "preview" in short or "exp" in short else 1,
            0 if any(ch.isdigit() for ch in short.split("-")[-1]) else 1)


def available_models(key: str) -> list[str]:
    """Every model this key may call generateContent on, best first."""
    client = _client(key)

    usable = []
    for model in client.models.list():
        name = (model.name or "").removeprefix("models/")
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" not in actions:
            continue
        if not name.startswith("gemini") or any(bad in name for bad in _EXCLUDE):
            continue
        usable.append(name)

    return sorted(usable, key=_model_rank, reverse=True)


@lru_cache(maxsize=4)
def resolve_model(key: str) -> str:
    """The model to call: an explicit override, else the best one the key can see.

    Cached per key - one extra list call per process, not per question.
    """
    override = None
    try:
        import streamlit as st
        if "GEMINI_MODEL" in st.secrets:
            override = str(st.secrets["GEMINI_MODEL"]).strip()
    except Exception:
        pass
    override = override or os.environ.get("GEMINI_MODEL", "").strip()
    if override:
        return override

    try:
        found = available_models(key)
    except Exception:
        return FALLBACK_MODEL  # can't list - let the call itself produce the real error
    return found[0] if found else FALLBACK_MODEL


def _config():
    from google.genai import types
    return types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[types.Tool(function_declarations=TOOLS)],
        temperature=0.2,  # this is a numbers job, not a writing one
    )


def history_from(turns: list[dict]):
    """Prior chat turns as Gemini contents, so follow-up questions keep their referent.

    Only the text is replayed - the tool results are not, since re-answering from a
    stale result is exactly the failure the tool loop exists to prevent. The model
    re-calls the tool when a follow-up needs the figures again.
    """
    from google.genai import types
    return [types.Content(role="user" if t["role"] == "user" else "model",
                          parts=[types.Part(text=t["content"])])
            for t in turns]


def ask(question: str, history: list | None = None, key: str | None = None):
    """One turn through the tool loop.

    Returns (answer_text, trace) where trace lists the tool calls made, so the UI
    can show its working - the audit trail is the point of the whole design.
    """
    from google.genai import types

    key = key or api_key()
    if not key:
        return ("No API key configured - see the README section 'Wiring the Gemini key'.", [])

    client = _client(key)
    model = resolve_model(key)
    contents = list(history or [])
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    trace = []
    for _ in range(MAX_TOOL_TURNS):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=_config())
        except Exception as exc:
            # A retired model 404s. Drop the cached choice, re-discover once, and
            # retry - otherwise every deployed session breaks on Google's timetable.
            if "NOT_FOUND" in str(exc) or "404" in str(exc):
                resolve_model.cache_clear()
                retry = resolve_model(key)
                if retry != model:
                    model = retry
                    continue
            return (f"The model call failed: {type(exc).__name__}: {exc}", trace)

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or not candidate.content or not candidate.content.parts:
            return ("The model returned nothing. Try rephrasing the question.", trace)

        calls = [p.function_call for p in candidate.content.parts if p.function_call]
        if not calls:
            return ((response.text or "").strip() or "The model returned an empty answer.", trace)

        contents.append(candidate.content)

        parts = []
        for call in calls:
            args = dict(call.args or {})
            fn = DISPATCH.get(call.name)
            try:
                result = jsonable(fn(**args)) if fn else {"error": f"Unknown tool '{call.name}'."}
            except Exception as exc:  # a bad argument must not kill the conversation
                result = {"error": f"{type(exc).__name__}: {exc}"}
            trace.append({"tool": call.name, "args": args, "result": result})
            parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))

        contents.append(types.Content(role="user", parts=parts))

    return (f"Stopped after {MAX_TOOL_TURNS} tool calls without settling on an answer. "
            f"Ask something narrower.", trace)


# --------------------------------------------------------------------------
# Wiring check - run this before trusting the deploy
# --------------------------------------------------------------------------

def check() -> int:
    """Prove the key resolves, reaches Google, and comes back through a real tool call."""
    print("Gemini wiring check\n" + "-" * 60)

    key = api_key()
    source = key_source()
    if not key:
        print("  key            NOT FOUND")
        print("\n  Set it one of two ways:")
        print("    .streamlit/secrets.toml   GEMINI_API_KEY = \"AIza...\"")
        print("    shell                     export GEMINI_API_KEY=AIza...")
        return 1

    print(f"  key            found via {source}, {len(key)} chars, starts {key[:6]}...")
    if not key.startswith("AIza"):
        print("  WARNING        Google AI Studio keys normally start 'AIza' - check you pasted the right value")

    try:
        usable = available_models(key)
        print(f"  auth           OK - {len(usable)} usable models visible to this key")
    except Exception as exc:
        print(f"  auth           FAILED - {type(exc).__name__}: {exc}")
        print("\n  A 400 with API_KEY_INVALID means the key is wrong or has an IP/referrer restriction.")
        print("  A 403 usually means the Generative Language API is not enabled on the key's project.")
        return 1

    if not usable:
        print("  models         FAILED - this key sees no Gemini model that supports generateContent")
        print("  Run `uv run python ai_analyst.py --models` to see the raw list.")
        return 1

    chosen = resolve_model(key)
    print(f"  model          {chosen}")
    print(f"                 next best: {', '.join(usable[1:4]) or 'none'}")
    if chosen not in usable:
        print(f"  WARNING        '{chosen}' came from GEMINI_MODEL but is not in this key's list")

    try:
        overview = portfolio_overview()
        print(f"  artifacts      OK - {overview['entities']} entities, "
              f"{overview['transactions']:,} transactions loaded")
    except Exception as exc:
        print(f"  artifacts      FAILED - {type(exc).__name__}: {exc}")
        print("  Run `uv run python build_artifacts.py` first, or check outputs/artifacts/ is present.")
        return 1

    answer, trace = ask("In one sentence, what share of reported financials does the bank carry?")
    if not trace:
        print("  tool loop      FAILED - the model answered without calling a tool")
        return 1

    print(f"  tool loop      OK - called {', '.join(t['tool'] for t in trace)}")
    print("-" * 60)
    print(f"  {answer}")
    return 0


def models_report() -> int:
    """Every model the key can see, so a 404 can be diagnosed rather than guessed at."""
    key = api_key()
    if not key:
        print("No API key configured.")
        return 1

    from google import genai
    client = genai.Client(api_key=key)

    rows = []
    for model in client.models.list():
        name = (model.name or "").removeprefix("models/")
        actions = getattr(model, "supported_actions", None) or []
        rows.append((name, "generateContent" in actions, ",".join(actions)))

    usable = available_models(key)
    print(f"{len(rows)} models visible, {len(usable)} usable for this panel\n")
    print(f"  chosen: {resolve_model(key)}\n")

    print("  usable, best first:")
    for name in usable:
        print(f"    {name}")

    print("\n  everything else:")
    for name, ok, actions in sorted(rows):
        if name not in usable:
            print(f"    {name:<45} {'generateContent' if ok else actions[:40]}")
    return 0


def main() -> int:
    if "--models" in sys.argv:
        return models_report()
    if "--check" in sys.argv:
        return check()
    if "--ask" in sys.argv:
        i = sys.argv.index("--ask")
        question = " ".join(sys.argv[i + 1:]).strip()
        if not question:
            print("Usage: uv run python ai_analyst.py --ask \"your question\"")
            return 1
        answer, trace = ask(question)
        for t in trace:
            print(f"  -> {t['tool']}({', '.join(f'{k}={v!r}' for k, v in t['args'].items())})")
        print("\n" + answer)
        return 0

    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
