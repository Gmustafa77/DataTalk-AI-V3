"""Supervisor agent — decides how every chat question is answered (no LLM calls itself).

Routing order for one question:

1. **Input check**      — empty / too short / too long → guidance message (0 LLM calls).
2. **Quick action**     — deterministic analysis on the in-memory dataset (0 LLM calls).
3. **Q&A store**        — normalised question found in ``QA.json`` → stored answer
   (0 LLM calls). Sits in front of every AI invocation and is provider-independent.
4. **Configuration**    — no ``GROQ_API_KEY`` → explanatory message (0 LLM calls).
5. **Entity resolution** — partial names, aliases, abbreviations and several provider
   names in one question are resolved to exact database values and added to the
   grounding values for this question only (0 LLM calls). Questions with no such
   mentions (or only exact names) keep their original grounding values unchanged.
6. **Analytics engine** — planner → SQL → insight → synthesis/validation (see README).
7. **Direct-SQL fallback** — only if the engine raised unexpectedly: one SQL generation
   (+0–1 repair) and the composer narrative (0–1).

Visualization plans
-------------------
For engine answers, every answered sub-question gets a deterministic
visualization plan (``core.viz_recommender``) built from its verified result
table plus the analytical context the SQL agent produced (question, SQL, grain,
dimensions). Plans are stored in ``AssistantAnswer.meta["viz"]`` and let the UI
switch chart types locally — switching never comes back to this class, so it
never triggers an LLM call or SQL. Other routes need no plan here: the UI derives
one from ``answer.table`` on demand.

A fully answered, aggregate AI result is written to ``QA.json`` for reuse. The store
is a backend concern only: a reused answer has the same route, content and layout as
a newly generated one, and nothing user-facing mentions it.

Every AI run is wrapped in :func:`core.llm.trace_calls`; the per-question call
accounting is returned in ``AssistantAnswer.meta`` for server-side telemetry.

Production guarantees of this revision
--------------------------------------
* ``Supervisor.answer`` never raises for a recoverable fault: every route and every
  optional step (fast path, store lookup/write, entity resolution, charts, fallback,
  status callback) degrades to the next route or to a plain message.
* Logs carry exception *types* and a short request id, never question text, SQL or
  exception messages (these can contain member data); full tracebacks only at DEBUG.
* Stored answers are stamped with the prompt version and a dataset fingerprint, and are
  ignored when either no longer matches.
* Only validated ``analytics_engine`` answers can be written to the store.
* Public names, signatures and user-facing texts are unchanged; new parameters are optional.
"""
from __future__ import annotations

import hashlib
import logging
import time
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import pandas as pd

import prompts as P
from agents.analytics_reasoning_agent import AnalyticsEngine, PipelineResult
from agents.insight_generation_agent import compose_insight
from agents.visualization_agent import ChartSource, fallback_charts
from core.entity_resolver import EntityResolverCache
from core.fast_paths import match_fast_path, run_fast_path
from core.llm import trace_calls
from core.plot_style import style_figure
from core.qa_store import (
    QAStore,
    answer_from_payload,
    answer_to_payload,
    depends_on_conversation,
    question_may_contain_pii,
)
from core.schemas import AssistantAnswer
from core.settings import SENSITIVE_COLUMNS, Settings
from core.sql_guard import projects_sensitive_columns
from core.viz_recommender import recommend_visualizations

log = logging.getLogger("datatalk.supervisor")

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 2000
EXECUTION_ERROR_MESSAGE = ("The query could not be completed against the live database. Please retry, or refine "
                           "the names, statuses or date ranges in your question.")
UNEXPECTED_ERROR_MESSAGE = ("The assistant could not complete this request. Please retry, or refine the names, "
                            "statuses or date ranges in your question.")

# Key added to the stored payload so a reused answer can be checked against the current
# prompt set and dataset. It is removed again before the payload is rebuilt into an answer.
_CACHE_STAMP_KEY = "_cache_stamp"


# ═══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _log_exc(message: str, exc: BaseException) -> None:
    """Log the exception TYPE only; the traceback (which may hold member data) only at DEBUG."""
    log.error("%s: %s", message, type(exc).__name__,
              exc_info=exc if log.isEnabledFor(logging.DEBUG) else None)


def _clean_question(question: Any) -> str:
    """Normalise whitespace and drop control / zero-width characters.

    Invisible characters would otherwise pass the minimum-length check and reach the LLM
    and the Q&A store key.
    """
    text = "" if question is None else str(question)
    text = "".join(ch for ch in text if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf"))
    return " ".join(text.split())


def _safe_status(status_fn: Callable[[str], None] | None) -> Callable[[str], None]:
    """Wrap the UI status callback so a UI glitch can never abort an answer."""
    def _emit(message: str) -> None:
        if status_fn is None:
            return
        try:
            status_fn(message)
        except Exception as exc:  # noqa: BLE001 - UI callback is external code
            log.debug("status callback failed: %s", type(exc).__name__)
    return _emit


@dataclass
class QuestionContext:
    """Everything about the current session the supervisor needs to answer a question."""

    base_df: pd.DataFrame            # full loaded dataset (fast paths run on this)
    available_values: dict           # grounding values for the planner / SQL agent
    conversation: list[str]          # recent chat messages (planner uses the last three)


class Supervisor:
    def __init__(self, settings: Settings, engine: AnalyticsEngine) -> None:
        self.settings = settings
        self.engine = engine
        # Optional components must never stop the app from starting.
        self.qa_store = None
        if settings.qa_cache_enabled:
            try:
                self.qa_store = QAStore.from_settings(settings)
            except Exception as exc:  # noqa: BLE001
                _log_exc("Q&A store unavailable; continuing without it", exc)
        # Partial / alias / multi-name provider matching (deterministic, no LLM calls).
        self.entity_resolver = None
        if getattr(settings, "entity_resolution_enabled", True):
            try:
                self.entity_resolver = EntityResolverCache.from_settings(settings)
            except Exception as exc:  # noqa: BLE001
                _log_exc("entity resolver unavailable; continuing without it", exc)

    # ── entry point ──────────────────────────────────────────────────────────────
    def answer(self, question: str, ctx: QuestionContext, status_fn: Callable[[str], None]) -> AssistantAnswer:
        t0 = time.perf_counter()
        rid = uuid.uuid4().hex[:8]
        emit = _safe_status(status_fn)
        q = _clean_question(question)
        if len(q) < MIN_QUESTION_CHARS:
            return self._done(AssistantAnswer(text="Please provide a more specific Appeals question.",
                                              route="rejected"), t0, rid)
        if len(q) > MAX_QUESTION_CHARS:
            return self._done(AssistantAnswer(text=f"Please shorten the question to under {MAX_QUESTION_CHARS} "
                                                   "characters.", route="rejected"), t0, rid)

        # Quick action. A failure here falls through to the next route instead of crashing the chat.
        try:
            match = match_fast_path(q, ctx.base_df)
        except Exception as exc:  # noqa: BLE001
            _log_exc(f"[{rid}] fast-path matching failed", exc)
            match = None
        if match is not None:
            emit("⚡ Running a deterministic quick action…")
            try:
                answer = run_fast_path(match, ctx.base_df, emit)
            except Exception as exc:  # noqa: BLE001
                _log_exc(f"[{rid}] fast path failed; continuing to the next route", exc)
            else:
                self._ensure_chart(answer, q)
                answer.meta.update(fast_path=match.name)
                return self._done(answer, t0, rid)

        # Q&A store: checked before any AI/LLM invocation.
        as_of = date.today().isoformat()
        reusable = self._is_reusable(q, ctx)
        if reusable:
            stored = self._from_store(q, as_of, ctx)
            if stored is not None:
                emit("Assembling the verified answer…")
                return self._done(stored, t0, rid)

        if not self.settings.llm_configured:
            return self._done(AssistantAnswer(
                text="⚠️ The AI assistant is not configured: set `GROQ_API_KEY` in `.env` and restart the app. "
                     "Quick actions and dashboards keep working.", route="rejected"), t0, rid)

        log.info("[%s] Q&A cache miss or bypass; starting AI workflow", rid)

        # Entity resolution: map partial names, aliases, abbreviations and multiple
        # provider references to exact database values for THIS question only.
        # Returns the original available_values object unchanged when nothing needs
        # resolving, so existing questions behave exactly as before.
        ctx = self._resolve_entities(q, ctx)

        answer = self._run_ai(q, ctx, (ctx.conversation or [])[-3:], emit, as_of)

        # Backend diagnostics only. Existing UI remains unchanged.
        answer.meta["qa_cache"] = self._persist(q, answer, ctx, as_of, reusable)

        log.info("[%s] Q&A cache outcome=%s, route=%s, llm_calls=%s", rid, answer.meta.get("qa_cache"),
                 answer.route, answer.meta.get("llm_calls", 0))
        return self._done(answer, t0, rid)

    # ── Q&A store ────────────────────────────────────────────────────────────────
    def _is_reusable(self, q: str, ctx: QuestionContext) -> bool:
        """True when a stored answer may be used for / saved from this question."""
        if self.qa_store is None:
            return False
        try:
            return not depends_on_conversation(q, ctx.conversation or [])
        except Exception as exc:  # noqa: BLE001
            _log_exc("conversation-dependence check failed; bypassing the Q&A store", exc)
            return False  # fail safe: never reuse or store when unsure

    @staticmethod
    def _dataset_fingerprint(df: pd.DataFrame) -> str:
        """Cheap schema + row-count fingerprint; catches a reloaded / re-extracted dataset."""
        try:
            raw = f"{tuple(df.shape)}|{'|'.join(map(str, df.columns))}"
            return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        except Exception:  # noqa: BLE001
            return ""

    def _cache_stamp(self, ctx: QuestionContext) -> dict:
        return {"prompt_version": P.PROMPT_VERSION, "dataset_fingerprint": self._dataset_fingerprint(ctx.base_df)}

    def _stale_reason(self, stamp: Any, ctx: QuestionContext) -> str | None:
        """Why a stored answer must not be reused, or ``None`` when it is still valid.

        Entries written before this revision carry no stamp and stay reusable (they expire
        with their ``as_of_date``); clear ``QA.json`` once after deploying to retire them early.
        """
        if not getattr(self.settings, "qa_cache_enforce_versions", True) or not isinstance(stamp, dict):
            return None
        stored_version = stamp.get("prompt_version")
        if stored_version is not None and stored_version != P.PROMPT_VERSION:
            return "prompt_version_changed"
        stored_fp = stamp.get("dataset_fingerprint")
        if stored_fp:
            current_fp = self._dataset_fingerprint(ctx.base_df)
            if current_fp and current_fp != stored_fp:
                return "dataset_changed"
        return None

    def _from_store(
        self,
        q: str,
        as_of: str,
        ctx: QuestionContext,
    ) -> AssistantAnswer | None:
        try:
            payload = self.qa_store.lookup(q, as_of_date=as_of)
        except Exception as exc:  # noqa: BLE001 - corrupt file, lock, disk error
            _log_exc("Q&A lookup failed; running the AI workflow", exc)
            return None
        if payload is None:
            return None

        stamp = None
        if isinstance(payload, dict) and _CACHE_STAMP_KEY in payload:
            stamp = payload[_CACHE_STAMP_KEY]
            payload = {k: v for k, v in payload.items() if k != _CACHE_STAMP_KEY}
        reason = self._stale_reason(stamp, ctx)
        if reason:
            log.info("stored answer ignored (%s); running the AI workflow", reason)
            return None

        try:
            answer = answer_from_payload(payload, ctx.base_df)
        except Exception as exc:  # noqa: BLE001
            _log_exc("stored answer could not be rebuilt; running the AI workflow", exc)
            return None
        answer.meta.update(
            llm_calls=0,
            llm_requests=0,
            calls_by_purpose={},
            tokens_in=0,
            tokens_out=0,
            est_cost_usd=0.0,
        )
        # Record that this answer came from the cache.
        answer.meta["qa_cache"] = "hit"
        log.info("question answered from the Q&A store (0 LLM calls)")
        return answer

    def _persist(self, q: str, answer: AssistantAnswer, ctx: QuestionContext, as_of: str, reusable: bool) -> str:
        """Save a reusable answer; returns the ``meta['qa_cache']`` diagnostic value."""
        if self.qa_store is None:
            return "disabled"
        if not reusable:
            return "bypass_conversation_context"
        # Only validated engine answers are reusable; fallback / quick-action answers never are.
        if not answer.cacheable or answer.route != "analytics_engine":
            return "bypass_result_not_cacheable"
        try:
            if question_may_contain_pii(q):
                return "bypass_sensitive_question"
            payload = answer_to_payload(answer)
            if payload is None:
                return "serialization_failed"
            if isinstance(payload, dict):
                payload = {**payload, _CACHE_STAMP_KEY: self._cache_stamp(ctx)}
            saved = self.qa_store.store(
                q,
                payload,
                as_of_date=as_of,
                metadata={
                    "model": getattr(self.settings, "groq_model", ""),
                    "prompt_version": P.PROMPT_VERSION,
                    "dataset_fingerprint": self._dataset_fingerprint(ctx.base_df),
                },
            )
            return "stored" if saved else "not_written"
        except Exception as exc:  # noqa: BLE001
            # A persistence problem must not discard an otherwise successful chatbot answer.
            _log_exc("Q&A persistence failed; returning the generated answer", exc)
            return "write_error"

    # ── entity resolution ────────────────────────────────────────────────────────
    def _resolve_entities(self, q: str, ctx: QuestionContext) -> QuestionContext:
        """Return a context whose grounding values include resolved provider names.

        The session's shared ``available_values`` is never mutated: a new context is
        returned only when resolution added something; otherwise ``ctx`` itself is
        returned. Any failure degrades silently to the original behaviour.
        """
        if self.entity_resolver is None:
            return ctx
        try:
            values = self.entity_resolver.augment(q, ctx.base_df, ctx.available_values)
        except Exception as exc:  # noqa: BLE001
            _log_exc("entity resolution failed; continuing with the original grounding values", exc)
            return ctx
        if values is ctx.available_values:
            return ctx
        return replace(ctx, available_values=values)

    # ── AI workflow ──────────────────────────────────────────────────────────────
    def _run_ai(self, q: str, ctx: QuestionContext, recent: list[str],
                status_fn: Callable[[str], None], as_of: str | None = None) -> AssistantAnswer:
        as_of = as_of or date.today().isoformat()
        with trace_calls() as trace:
            try:
                result = self.engine.run(q, available_values=ctx.available_values, status_fn=status_fn,
                                         conversation_context=recent)
                answer = self._from_pipeline(result, ctx)
            except Exception as exc:  # unexpected engine failure → degrade, never crash the chat
                _log_exc("analytics engine failed; using direct-SQL fallback", exc)
                status_fn("Primary pipeline unavailable; using the direct query path…")
                answer = self._safe_direct_sql_fallback(q, ctx, type(exc).__name__, as_of)
        try:
            answer.meta.update(trace.summary())
        except Exception as exc:  # noqa: BLE001 - telemetry must not break the answer
            _log_exc("call-trace summary failed", exc)
        self._ensure_chart(answer, q)
        return answer

    def _from_pipeline(self, result: PipelineResult, ctx: QuestionContext) -> AssistantAnswer:
        outcomes = result.outcomes or []
        validation = result.validation or {}
        sensitive = any(
            o.contract is not None and (not o.contract.is_aggregate
                                        or projects_sensitive_columns(o.contract.sql, SENSITIVE_COLUMNS))
            for o in outcomes)
        meta: dict = {"plan_status": result.plan_status, "question_count": result.question_count,
                      "timings": result.timings, "validation": {k: v for k, v in validation.items()
                                                                if k in ("verdict", "unanswered_ids",
                                                                         "dropped_numeric_claims",
                                                                         "charts_reconciled")},
                      "defects": len(validation.get("defects") or [])}
        viz = self._viz_entries(result)
        if viz:
            meta["viz"] = viz
        return AssistantAnswer(text=result.text, table=result.primary_df, figures=result.figures,
                               insight=result.insight_card or None, filtered_df=ctx.base_df,
                               route="analytics_engine", cacheable=result.fully_answered and not sensitive,
                               meta=meta)

    @staticmethod
    def _viz_entries(result: PipelineResult) -> list[dict]:
        """Deterministic visualization plan per answered sub-question (no LLM calls).

        The primary result table is already on the answer, so its entry stores a
        ``primary`` flag instead of a second copy of the data.
        """
        entries: list[dict] = []
        try:
            for o in result.outcomes or []:
                c = o.contract
                if o.status != "ANSWERED" or c is None or c.df is None or c.df.empty:
                    continue
                plan = recommend_visualizations(c.df, {
                    "question": c.question_text, "sql": c.sql, "grain": c.grain,
                    "dimensions": list(c.dimensions or []), "is_aggregate": c.is_aggregate,
                    "analysis_type": "drilldown" if not c.is_aggregate else "",
                })
                if not plan.compatible_charts:
                    continue
                is_primary = result.primary_df is not None and c.df is result.primary_df
                entries.append({"id": c.sub_question_id, "title": c.question_text, "plan": plan.to_dict(),
                                "primary": is_primary, "table": None if is_primary else c.df})
        except Exception as exc:  # charts are optional; the answer must never fail because of them
            _log_exc("visualization plans could not be built", exc)
            return []
        return entries

    def _safe_direct_sql_fallback(self, q: str, ctx: QuestionContext, error_type: str, as_of: str) -> AssistantAnswer:
        """The fallback is the last line of defence, so it must never raise."""
        try:
            return self._direct_sql_fallback(q, ctx, error_type, as_of)
        except Exception as exc:  # noqa: BLE001
            _log_exc("direct-SQL fallback failed", exc)
            return AssistantAnswer(text=UNEXPECTED_ERROR_MESSAGE, route="direct_sql_fallback", cacheable=False,
                                   meta={"engine_error": error_type, "fallback_error": type(exc).__name__})

    def _direct_sql_fallback(self, q: str, ctx: QuestionContext, error_type: str,
                             as_of_date: str | None = None) -> AssistantAnswer:
        sub = {"id": "q1", "question_text": q, "answerable": True, "analysis_type": "metric", "depends_on": None}
        out = self.engine.sql_agent.generate(sub, as_of_date=as_of_date or date.today().isoformat(),
                                             available_values=ctx.available_values)
        meta = {"engine_error": error_type}
        if not out.ok:
            return AssistantAnswer(text=out.failure_reason or "The query could not be completed.",
                                   route="direct_sql_fallback", cacheable=False, meta=meta)
        try:
            df, _ = self.engine.run_sql(out.check.sql)
        except Exception as exc:
            log.warning("fallback sql execution failed: %s", type(exc).__name__)
            return AssistantAnswer(text=EXECUTION_ERROR_MESSAGE, route="direct_sql_fallback", cacheable=False,
                                   meta=meta)
        if df.empty:
            return AssistantAnswer(text="No records matched the criteria. Verify names, statuses and date ranges.",
                                   table=df, route="direct_sql_fallback", cacheable=False, meta=meta)
        try:
            insight = compose_insight(df, question=q, sql=out.check.sql, config=self.engine.composer_cfg,
                                      llm=self.engine.llm if self.settings.insight_llm_narrative else None)
            insight["_sql"] = out.check.sql
        except Exception as exc:  # the verified table is still worth showing
            _log_exc("fallback insight composition failed", exc)
            insight = {"insight": "The query completed. The verified result table is shown below.",
                       "_sql": out.check.sql}
        try:
            figures, _ = self.engine.charts.build(ChartSource("q1", df, ""))
        except Exception as exc:
            _log_exc("fallback chart build failed", exc)
            figures = []
        return AssistantAnswer(text=insight.get("insight", ""), table=df, figures=figures, insight=insight,
                               filtered_df=ctx.base_df, route="direct_sql_fallback", cacheable=False, meta=meta)

    # ── helpers ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_chart(answer: AssistantAnswer, question: str) -> None:
        """Add a fallback chart when a table arrived without figures."""
        if answer.figures or answer.table is None or answer.table.empty:
            return
        try:
            answer.figures = fallback_charts(answer.table, question, style_fn=lambda f: style_figure(f, 440))
        except Exception as exc:  # a chart problem must never discard a verified answer
            _log_exc("fallback chart could not be built", exc)
            return
        if not answer.figures:
            answer.text += ("\n\n_No reliable chart type was available for this result shape; "
                            "the verified table is shown instead._")

    @staticmethod
    def _done(answer: AssistantAnswer, t0: float, request_id: str = "") -> AssistantAnswer:
        answer.meta.setdefault("llm_calls", 0)
        answer.meta["route"] = answer.route
        if request_id:
            answer.meta["request_id"] = request_id
        answer.meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return answer