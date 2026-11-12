"""Analytics reasoning agent — orchestrates the open-question AI workflow.

    question
      → PlannerAgent           (LLM: planner, 1 call)
      → sub-questions scheduled as a DAG, wave by wave (every question whose
        dependencies are done runs in parallel):
           SqlAgent            (LLM: sql_generation 1, sql_repair 0–1)
           read-only validator + SqlExecutor (deterministic)
             └ on a database error: SqlAgent.repair_after_execution_error
                                    (LLM: execution_repair 0–max_execution_repairs)
           ChartBuilder        (deterministic, reconciled to the table)
           compose_insight     (LLM: insight_narrative 0–1)
      → ResponseAgent          (LLM: synthesis 1 + response_validation 0–1, only for ≥2 sub-questions)
      → PipelineResult

Exact call counts are documented in the README ("LLM Usage and Call Accounting").
This module has no Streamlit imports; every dependency is injected so the engine
can be tested with a fake LLM and a fake SQL executor.

Changes vs. the previous version
  1. Execution errors are fed back to the SQL agent for a bounded repair.
  2. A dependent question whose upstream failed is NOT_ANSWERED with a clear reason.
  3. ``depends_on`` may be a single id (legacy) or a list of ids.
  4. Parallelism is on by default; SQL is serialised with a lock unless the
     executor is declared thread-safe (``sql_thread_safe=True``).
  5. Dependent questions are scheduled in topological waves, not one by one.

Production hardening in this revision (every new option defaults to the old behaviour,
except where stated)
  * Optional wall-clock budget for the execution stage (``request_timeout_seconds``) and for
    waiting on the SQL lock (``sql_lock_timeout_seconds``); a stuck query can no longer block
    the request, or every other session, forever.
  * Plan ids are normalised (missing / duplicate ids can no longer crash the run).
  * Text for a question that FAILED is always the deterministic failure reason — it is never
    model-written, so it cannot contain unverified claims. A synthesised answer whose status
    contradicts the verified outcome is replaced by the verified text.
  * The QA validator's verdict is honoured: a FAIL is never cacheable, blocking defects are no
    longer lost to the 5-item cut-off, and (``fallback_on_blocking_defect``, default on) a
    blocking defect replaces the synthesised text with the deterministic per-question text.
  * The validator receives the verified evidence (``evidence=``), so it can check claims.
  * Row-level upstream results never pass sensitive columns to the LLM.
  * A result that hit the row limit is flagged to the user as possibly incomplete.
  * Logs carry exception types only (exception text can contain member data).
"""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

import prompts as P
from agents.insight_generation_agent import ComposerConfig, compose_insight, verify_text
from agents.planner_agent import PlannerAgent
from agents.response_agent import ResponseAgent
from agents.sql_agent import SqlAgent
from agents.visualization_agent import ChartBuilder, ChartConfig
from core.llm import StructuredLLM
from core.settings import SENSITIVE_COLUMNS
from core.sql_guard import SqlCheck

log = logging.getLogger("datatalk.engine")

UNRELATED_MESSAGE = (
    "No matching data was found for your request. Your question does not appear to be related to the "
    "available Appeals dataset. Please ask about appeal status trends, open vs closed appeals, turnaround "
    "time, outcomes, providers, members, date ranges, denials, or overturn analysis."
)
TIMED_OUT_MESSAGE = ("This question took too long to complete and was stopped. Please retry, or narrow the "
                     "date range or filters.")

# Errors that a rewritten query cannot fix; retrying the SQL through the LLM would waste a call.
_NON_REPAIRABLE = (TimeoutError, ConnectionError, PermissionError)


def _log_exc(message: str, exc: BaseException) -> None:
    """Log the exception TYPE only; the traceback (which may hold member data) only at DEBUG."""
    log.error("%s: %s", message, type(exc).__name__,
              exc_info=exc if log.isEnabledFor(logging.DEBUG) else None)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration and contracts
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class EngineConfig:
    fq_table: str
    raw_columns: list
    derived_columns: set
    sla_days: int = 30
    escalation_level: int = 2
    max_row_query_limit: int = 200
    max_result_rows: int = 5000
    max_sub_questions: int = 5
    max_parallel_workers: int = 3
    sql_thread_safe: bool = False          # False → run_sql calls are serialised with a lock
    enable_sql_repair: bool = True
    max_execution_repairs: int = 1         # LLM retries after a database error (0 disables)
    enable_response_validator: bool = True
    enable_llm_narrative: bool = True
    polarity_overrides: dict = field(default_factory=dict)   # {"Column": "lower"|"higher"|"neutral"}
    min_denominator_for_rate: int = 5
    charts: ChartConfig = field(default_factory=ChartConfig)
    # ── added (all optional) ────────────────────────────────────────────────────
    column_types: dict | None = None       # {"Column": "string"|"date"|...}; given to the SQL agent
    request_timeout_seconds: float = 0.0   # execution-stage budget; 0 = unlimited (recommended: 120)
    sql_lock_timeout_seconds: float = 0.0  # max wait for the serialised SQL executor; 0 = unlimited
    fallback_on_blocking_defect: bool = True   # QA gate: blocking defect → deterministic text


@dataclass
class ResultContract:
    """Immutable evidence package. Everything downstream reads only from here."""

    sub_question_id: str
    question_text: str
    sql: str
    intent: str
    df: pd.DataFrame
    grain: str
    dimensions: list
    measures: list
    row_count: int
    truncated: bool
    result_hash: str
    latency_ms: float
    is_aggregate: bool = True
    from_cache: bool = False

    @staticmethod
    def hash_df(df: pd.DataFrame) -> str:
        try:
            payload = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values
            return hashlib.sha256(payload.tobytes()).hexdigest()[:16]
        except TypeError:  # unhashable cell values
            return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]


@dataclass
class SubQuestionOutcome:
    id: str
    question_text: str
    status: str                      # ANSWERED | NOT_ANSWERED
    failure_reason: str = ""
    contract: ResultContract | None = None
    insight: dict = field(default_factory=dict)
    figures: list = field(default_factory=list)
    chart_notes: list = field(default_factory=list)
    dropped_claims: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    headline: str
    outcomes: list
    narrative: str
    primary_df: pd.DataFrame | None
    figures: list
    insight_card: dict
    validation: dict
    timings: dict
    question_count: int
    plan_status: str = "SUPPORTED"

    @property
    def text(self) -> str:
        if self.question_count > 1 and self.headline:
            return f"**{self.headline}**\n\n{self.narrative}"
        return self.narrative or self.headline

    @property
    def fully_answered(self) -> bool:
        return (self.question_count > 0 and bool(self.outcomes)
                and all(o.status == "ANSWERED" for o in self.outcomes)
                and self.validation.get("verdict", "PASS") == "PASS"
                and not self.validation.get("unanswered_ids")
                and not any(d.get("severity") == "blocking" for d in self.validation.get("defects", [])))


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def dep_ids(sub: dict) -> list[str]:
    """Normalise ``depends_on`` (None | "q1" | ["q1", "q2"]) to a de-duplicated list of ids."""
    raw = sub.get("depends_on")
    if not raw:
        return []
    items = [raw] if isinstance(raw, str) else list(raw)
    out: list[str] = []
    for item in items:
        item = str(item).strip()
        if item and item not in out:
            out.append(item)
    return out


def _safe_db_error(exc: Exception, limit: int = 500) -> str:
    """Short, single-line database error for the repair prompt. Never shown to the end user."""
    return " ".join(str(exc).split())[:limit] or type(exc).__name__


# ═══════════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════════
class AnalyticsEngine:
    """Injected dependencies:

    * ``llm``       — :class:`core.llm.StructuredLLM`
    * ``validate``  — ``sql -> SqlCheck`` (read-only allow-list validator)
    * ``run_sql``   — ``sql -> (DataFrame, from_cache)``
    """

    def __init__(self, cfg: EngineConfig, llm: StructuredLLM, validate: Callable[[str], SqlCheck],
                 run_sql: Callable[[str], tuple[pd.DataFrame, bool]]) -> None:
        self.cfg = cfg
        self.run_sql = self._maybe_serialise(run_sql, cfg)
        common = dict(fq_table=cfg.fq_table, raw_columns=cfg.raw_columns, derived_columns=cfg.derived_columns,
                      sla_days=cfg.sla_days, escalation_level=cfg.escalation_level)
        self.planner = PlannerAgent(llm, **common, max_sub_questions=cfg.max_sub_questions)
        self.sql_agent = SqlAgent(llm, validate, **common, max_row_query_limit=cfg.max_row_query_limit,
                                  enable_repair=cfg.enable_sql_repair, column_types=cfg.column_types)
        self.responder = ResponseAgent(llm, enable_validator=cfg.enable_response_validator)
        # Older ResponseAgent versions have no `evidence` argument; detect once instead of guessing per call.
        try:
            self._review_takes_evidence = "evidence" in inspect.signature(self.responder.review).parameters
        except (TypeError, ValueError):
            self._review_takes_evidence = False
        self.charts = ChartBuilder(cfg.charts)
        self.llm = llm
        self.composer_cfg = ComposerConfig(sla_days=cfg.sla_days, min_group_volume=cfg.min_denominator_for_rate,
                                           polarity_overrides=dict(cfg.polarity_overrides or {}),
                                           enable_llm_narrative=cfg.enable_llm_narrative)

    @staticmethod
    def _maybe_serialise(run_sql: Callable, cfg: EngineConfig) -> Callable:
        """LLM calls run in parallel; SQL calls are serialised unless the executor says it is thread-safe."""
        if cfg.max_parallel_workers <= 1 or cfg.sql_thread_safe:
            return run_sql
        lock = threading.Lock()
        wait = float(cfg.sql_lock_timeout_seconds or 0)

        def locked(sql: str):
            if wait > 0:  # never let one stuck query block every other request forever
                if not lock.acquire(timeout=wait):
                    raise TimeoutError("timed out waiting for the SQL executor")
                try:
                    return run_sql(sql)
                finally:
                    lock.release()
            with lock:
                return run_sql(sql)
        return locked

    @staticmethod
    def _normalise_plan(subs: list[dict]) -> list[dict]:
        """Give every sub-question a unique, non-empty id so scheduling and synthesis cannot break."""
        seen: set[str] = set()
        out: list[dict] = []
        for position, sub in enumerate(subs, 1):
            sub = dict(sub)
            base = str(sub.get("id") or "").strip() or f"q{position}"
            sid, n = base, 2
            while sid in seen:
                sid, n = f"{base}_{n}", n + 1
            seen.add(sid)
            sub["id"] = sid
            out.append(sub)
        return out

    # ── public entry point ───────────────────────────────────────────────────────
    def run(self, user_query: str, *, available_values: dict, status_fn: Callable[[str], None] | None = None,
            conversation_context: list[str] | None = None, as_of_date: str | None = None) -> PipelineResult:
        t_start = time.perf_counter()
        timings: dict[str, Any] = {"prompt_version": P.PROMPT_VERSION}
        say = status_fn or (lambda _m: None)
        as_of = as_of_date or date.today().isoformat()

        say("Understanding the question and checking for multiple intents…")
        t0 = time.perf_counter()
        plan_status, subs = self.planner.plan(user_query, as_of_date=as_of, available_values=available_values,
                                              conversation_context=conversation_context)
        timings["plan_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if plan_status == "UNRELATED" or not subs:
            return PipelineResult(UNRELATED_MESSAGE, [], UNRELATED_MESSAGE, None, [], {},
                                  {"verdict": "PASS", "defects": [], "unanswered_ids": []}, timings, 0,
                                  plan_status="UNRELATED")
        subs = self._normalise_plan(subs)
        if len(subs) > 1:
            say(f"Detected {len(subs)} distinct questions — planning execution…")

        t0 = time.perf_counter()
        outcomes = self._execute_plan(subs, available_values, as_of, say)
        timings["execute_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        say("Assembling the verified answer…")
        t0 = time.perf_counter()
        headline, narrative = self._synthesize(user_query, outcomes)
        validation = self._validate_response(user_query, subs, outcomes, headline, narrative)
        if (self.cfg.fallback_on_blocking_defect and len(subs) > 1
                and any(d.get("severity") == "blocking" for d in validation.get("defects", []))):
            # The QA gate found a blocking problem in the synthesised text: use the verified
            # per-question text instead (the same fallback used for unverifiable numbers).
            headline, narrative = self._deterministic_synthesis(outcomes)
            validation["deterministic_fallback"] = True
        if validation.get("unanswered_ids"):
            narrative += self._gap_disclosure(validation["unanswered_ids"], outcomes)
        timings["synthesis_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

        answered = [o for o in outcomes if o.status == "ANSWERED" and o.contract is not None]
        return PipelineResult(
            headline=headline, outcomes=outcomes, narrative=narrative,
            primary_df=answered[0].contract.df if answered else None,
            figures=[f for o in outcomes for f in o.figures],
            insight_card=self._build_insight_card(outcomes, headline), validation=validation,
            timings=timings, question_count=len(subs), plan_status=plan_status,
        )

    # ── DAG execution (topological waves) ────────────────────────────────────────
    def _execute_plan(self, subs: list[dict], values: dict, as_of: str, say: Callable) -> list[SubQuestionOutcome]:
        """Run every question whose dependencies are finished, in parallel, until none are left.

        * unknown / self / circular dependencies → NOT_ANSWERED with a reason
        * a dependency that did not answer      → NOT_ANSWERED, no SQL generated, no LLM call spent
        * the optional time budget expires       → unfinished questions are NOT_ANSWERED (timed out)
        """
        known_ids = {s["id"] for s in subs}
        pending: dict[str, dict] = {s["id"]: s for s in subs}
        results: dict[str, SubQuestionOutcome] = {}
        finished = 0
        workers = max(1, min(self.cfg.max_parallel_workers, len(subs)))
        budget = float(self.cfg.request_timeout_seconds or 0)
        deadline = time.monotonic() + budget if budget > 0 else None

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            while pending:
                if deadline is not None and time.monotonic() >= deadline:
                    for sid, sub in pending.items():
                        results[sid] = self._timed_out(sub)
                    pending.clear()
                    break

                progressed = False
                runnable: list[dict] = []

                for sid, sub in list(pending.items()):
                    deps = dep_ids(sub)
                    bad = [d for d in deps if d not in known_ids or d == sid]
                    if bad:
                        results[sid] = self._not_answered(
                            sub, f"This question depends on {', '.join(bad)}, which is not part of the plan.")
                    elif all(d in results for d in deps):
                        failed = [d for d in deps if not self._is_answered(results[d])]
                        if failed:
                            results[sid] = self._upstream_failed(sub, failed, results)
                        else:
                            runnable.append(sub)
                            del pending[sid]
                            progressed = True
                            continue
                    else:
                        continue  # still waiting on an unfinished dependency
                    del pending[sid]
                    progressed = True
                    finished += 1
                    say(f"Completed {finished}/{len(subs)} question(s)…")

                if not progressed:  # nothing ready and nothing resolvable → circular dependency
                    for sid, sub in pending.items():
                        results[sid] = self._not_answered(sub, "This question has a circular dependency "
                                                               "on another question in the plan.")
                    break

                snapshot = dict(results)  # workers read upstream results; the main thread owns `results`
                futures = {pool.submit(contextvars.copy_context().run, self._run_sub_question,
                                       s, values, as_of, snapshot): s for s in runnable}
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    for fut in as_completed(futures, timeout=remaining):
                        s = futures[fut]
                        try:
                            results[s["id"]] = fut.result()
                        except Exception as exc:  # one branch must never kill the request
                            _log_exc(f"sub-question {s.get('id')} crashed", exc)
                            results[s["id"]] = self._crashed(s, exc)
                        finished += 1
                        say(f"Completed {finished}/{len(subs)} question(s)…")
                except FuturesTimeout:
                    # Threads cannot be killed: cancel what has not started and stop waiting.
                    for fut, s in futures.items():
                        if s["id"] not in results:
                            fut.cancel()
                            results[s["id"]] = self._timed_out(s)
                            finished += 1
                    for sid, sub in pending.items():
                        results[sid] = self._timed_out(sub)
                    pending.clear()
                    log.warning("execution stage stopped after %.0fs budget", budget)
                    break
        finally:
            # Without a budget keep the original semantics (wait for every worker to finish).
            pool.shutdown(wait=deadline is None, cancel_futures=True)

        return [results[s["id"]] for s in subs if s["id"] in results]

    @staticmethod
    def _is_answered(o: SubQuestionOutcome) -> bool:
        return o.status == "ANSWERED" and o.contract is not None

    @staticmethod
    def _not_answered(s: dict, reason: str) -> SubQuestionOutcome:
        return SubQuestionOutcome(id=s.get("id", "q?"), question_text=s.get("question_text", ""),
                                  status="NOT_ANSWERED", failure_reason=reason)

    @classmethod
    def _timed_out(cls, s: dict) -> SubQuestionOutcome:
        out = cls._not_answered(s, TIMED_OUT_MESSAGE)
        out.diagnostics = {"timed_out": True}
        return out

    @classmethod
    def _upstream_failed(cls, s: dict, failed: list[str], results: dict) -> SubQuestionOutcome:
        parts = [f"“{results[d].question_text.strip() or d}”" for d in failed]
        reason = ("This question builds on " + ", ".join(parts) + ", which could not be answered, "
                  "so it was not attempted.")
        out = cls._not_answered(s, reason)
        out.diagnostics = {"blocked_by": failed}
        return out

    @staticmethod
    def _crashed(s: dict, exc: Exception) -> SubQuestionOutcome:
        return SubQuestionOutcome(id=s.get("id", "q?"), question_text=s.get("question_text", ""),
                                  status="NOT_ANSWERED",
                                  failure_reason="The analytics service could not complete this question. "
                                                 "Please retry the request.",
                                  diagnostics={"error_type": type(exc).__name__})

    # ── one sub-question ─────────────────────────────────────────────────────────
    @staticmethod
    def _head_rows(up: SubQuestionOutcome, n: int = 10) -> list[dict]:
        if up.contract is None:
            return []
        head = up.contract.df.head(n).copy()
        if not up.contract.is_aggregate:
            # Row-level upstream results can carry member data; the next query only needs the
            # business keys, so never send sensitive columns to the LLM provider.
            sensitive = {str(c).strip().lower() for c in SENSITIVE_COLUMNS}
            head = head.drop(columns=[c for c in head.columns if str(c).strip().lower() in sensitive])
        for column in head.columns:
            if pd.api.types.is_datetime64_any_dtype(head[column]):
                head[column] = head[column].astype(str)
        return head.astype(object).where(pd.notna(head), None).to_dict("records")

    def _upstream_payload(self, sub: dict, upstream: dict | None) -> list[dict] | dict:
        """One dependency → flat row list (unchanged legacy shape). Several → ``{dep_id: rows}``."""
        deps = dep_ids(sub)
        if not deps or not upstream:
            return {} if len(deps) > 1 else []
        rows = {d: self._head_rows(upstream[d]) for d in deps if d in upstream}
        return rows.get(deps[0], []) if len(deps) == 1 else rows

    def _execute_sql(self, sub: dict, sql_out, values: dict, as_of: str, upstream_rows):
        """Run the validated SQL; on a database error give the LLM the error for a bounded repair.

        Returns ``(df, from_cache, sql_out, latency_ms, repairs_used, error)``.
        """
        qid = sub.get("id", "q?")
        repairs = 0
        while True:
            sql = sql_out.check.sql
            t0 = time.perf_counter()
            try:
                df, from_cache = self.run_sql(sql)
                return df, from_cache, sql_out, (time.perf_counter() - t0) * 1000, repairs, None
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000
                log.warning("sql execution failed for %s (repair %d/%d): %s", qid, repairs,
                            self.cfg.max_execution_repairs, type(exc).__name__)
                if (repairs >= self.cfg.max_execution_repairs or isinstance(exc, _NON_REPAIRABLE)):
                    return None, False, sql_out, latency, repairs, exc
                repairs += 1
                try:
                    fixed = self.sql_agent.repair_after_execution_error(
                        sub, failed_sql=sql, error=_safe_db_error(exc), as_of_date=as_of,
                        available_values=values, upstream_results=upstream_rows,
                        original=sql_out.generation)
                except Exception as repair_exc:
                    _log_exc(f"execution repair crashed for {qid}", repair_exc)
                    return None, False, sql_out, latency, repairs, exc
                # An unusable or identical rewrite will fail the same way — stop here.
                if not fixed.ok or fixed.check.sql.strip() == sql.strip():
                    return None, False, sql_out, latency, repairs, exc
                sql_out = fixed

    def _run_sub_question(self, sub: dict, values: dict, as_of: str,
                          upstream: dict | None = None) -> SubQuestionOutcome:
        qid, qtext = sub.get("id", "q1"), sub.get("question_text", "")
        if not sub.get("answerable", True):
            return SubQuestionOutcome(qid, qtext, "NOT_ANSWERED",
                                      sub.get("reason") or "This question is not supported by the Appeals dataset.")

        upstream_rows = self._upstream_payload(sub, upstream)
        sql_out = self.sql_agent.generate(sub, as_of_date=as_of, available_values=values,
                                          upstream_results=upstream_rows)
        if not sql_out.ok:
            return SubQuestionOutcome(qid, qtext, "NOT_ANSWERED", sql_out.failure_reason,
                                      diagnostics=sql_out.diagnostics)

        df, from_cache, sql_out, latency, repairs, error = self._execute_sql(
            sub, sql_out, values, as_of, upstream_rows)
        gen, check = sql_out.generation, sql_out.check
        sql = check.sql
        diag = {**sql_out.diagnostics, "sql": sql, "execution_repairs": repairs}

        if error is not None:  # Databricks errors are reported, not raised
            diag["error_type"] = type(error).__name__
            return SubQuestionOutcome(qid, qtext, "NOT_ANSWERED",
                                      "The validated query could not be completed against the live database. "
                                      "Please retry, or refine the names, statuses or date ranges.",
                                      diagnostics=diag)
        if df is None or df.empty:
            return SubQuestionOutcome(qid, qtext, "NOT_ANSWERED",
                                      "No records matched the criteria for this question. "
                                      "Verify names, statuses, and date ranges.", diagnostics=diag)

        truncated = len(df) >= self.cfg.max_result_rows
        df, norm_notes = self._normalize_result(df)
        if truncated:
            norm_notes.append(f"The result reached the {self.cfg.max_result_rows:,}-row limit and may be "
                              "incomplete; totals and rankings may be understated.")
        contract = ResultContract(
            sub_question_id=qid, question_text=qtext, sql=sql, intent=gen.intent, df=df,
            grain=str(gen.grain), dimensions=list(gen.dimensions),
            measures=[m.model_dump() if hasattr(m, "model_dump") else m for m in gen.measures],
            row_count=len(df), truncated=truncated,
            result_hash=ResultContract.hash_df(df), latency_ms=latency,
            is_aggregate=check.is_aggregate, from_cache=from_cache,
        )
        figures, notes = self.charts.build(contract, sub.get("analysis_type", ""))
        insight = self._compose(contract, notes + norm_notes)
        return SubQuestionOutcome(
            qid, qtext, "ANSWERED", contract=contract, insight=insight, figures=figures,
            chart_notes=notes + norm_notes, dropped_claims=insight.get("_dropped_claims", []),
            diagnostics={**diag, "latency_ms": round(latency, 1), "from_cache": from_cache,
                         "result_hash": contract.result_hash, "narrative_source": insight.get("narrative_source")},
        )

    def _compose(self, c: ResultContract, notes: list) -> dict:
        try:
            insight = compose_insight(c.df, question=c.question_text, sql=c.sql, config=self.composer_cfg,
                                      llm=self.llm if self.cfg.enable_llm_narrative else None, chart_notes=notes)
        except Exception as exc:  # the answer must not fail because the narrative did
            _log_exc(f"insight composition failed for {c.sub_question_id}", exc)
            msg = f"The validated query returned {c.row_count:,} verified row(s). Review the result table below."
            insight = {"insight": msg, "key_finding": {"text": msg, "refs": []}, "detailed_analysis": [],
                       "business_impact": [], "reasoning_steps": [], "evidence": [], "supporting_metrics": [],
                       "recommended_actions": [], "data_caveats": notes, "grounded": False,
                       "_dropped_claims": [], "_allowed_numbers": [], "composer_version": "fallback"}
        insight["question"] = c.question_text
        insight["_sql"] = c.sql
        insight["_result_hash"] = c.result_hash
        return insight

    @staticmethod
    def _normalize_result(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """Trim column names and drop exact duplicates. Never sums averages or rates."""
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        notes = []
        before = len(out)
        try:
            out = out.drop_duplicates()
        except TypeError:  # unhashable cells (arrays / maps): keep the rows exactly as returned
            pass
        if len(out) < before:
            notes.append(f"{before - len(out)} exact duplicate row(s) were removed from the result.")
        dim_like = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
        try:
            if dim_like and out.duplicated(subset=dim_like).any():
                notes.append("Some groups appear more than once in the result; values are shown exactly as returned.")
        except TypeError:
            pass
        return out.reset_index(drop=True), notes

    # ── synthesis ────────────────────────────────────────────────────────────────
    @staticmethod
    def _fallback_body(o: SubQuestionOutcome) -> str:
        """Deterministic, verified text for one question: the insight, or the failure reason."""
        return o.insight.get("insight", "") if o.status == "ANSWERED" else o.failure_reason

    @staticmethod
    def _outcomes_payload(outcomes: list[SubQuestionOutcome]) -> list[dict]:
        return [{
            "id": o.id, "question_text": o.question_text, "status": o.status,
            "failure_reason": o.failure_reason, "insight": o.insight.get("insight", ""),
            "key_facts": [{"label": e.get("label"), "value": e.get("display")}
                          for e in (o.insight.get("evidence") or [])[:12]],
        } for o in outcomes]

    def _deterministic_synthesis(self, outcomes: list[SubQuestionOutcome]) -> tuple[str, str]:
        lines = [f"**{i}. {o.question_text.strip()}**\n\n{self._fallback_body(o)}" for i, o in enumerate(outcomes, 1)]
        first = next((o for o in outcomes if o.status == "ANSWERED"), None)
        headline = first.insight.get("insight", "") if first else "Unable to answer."
        return headline, "\n\n".join(lines)

    def _synthesize(self, user_query: str, outcomes: list[SubQuestionOutcome]) -> tuple[str, str]:
        if not outcomes:
            return "Unable to answer.", "The request could not be completed. Please retry."
        if len(outcomes) == 1:
            o = outcomes[0]
            if o.status == "ANSWERED":
                kf = o.insight.get("insight", "")
                if not kf and o.contract is not None:
                    kf = (f"The validated query returned {o.contract.row_count:,} verified row(s). "
                          "Review the result table below.")
                return kf, kf
            return "Unable to answer.", o.failure_reason

        res = self.responder.synthesize(user_query, self._outcomes_payload(outcomes))

        # Every synthesised number must exist in the evidence of some answered question.
        allowed = sorted({tuple(a) for o in outcomes for a in (o.insight.get("_allowed_numbers") or [])}, key=repr)
        by_id = {a.id: a for a in (res.answers if res else [])}
        lines = []
        for i, o in enumerate(outcomes, 1):
            fallback = self._fallback_body(o)
            if o.status == "ANSWERED":
                cand = by_id.get(o.id)
                # A synthesised answer that says "not answered" contradicts the verified outcome.
                body = cand.answer if cand and cand.answer and cand.status != "NOT_ANSWERED" else fallback
                if body != fallback and not verify_text(body, allowed):
                    body = fallback
            else:
                body = fallback  # failure text is deterministic: never model-written, never unverified
            lines.append(f"**{i}. {o.question_text.strip()}**\n\n{body}")
        headline = res.headline if res else ""
        if headline and not verify_text(headline, allowed):
            headline = ""
        if not headline:
            first = next((o for o in outcomes if o.status == "ANSWERED"), None)
            headline = first.insight.get("insight", "") if first else "Unable to answer."
        note = res.cross_cutting_note if res else ""
        if note and not verify_text(note, allowed):
            note = ""
        return headline, "\n\n".join(lines) + (f"\n\n_{note}_" if note else "")

    @staticmethod
    def _charts_reconciled(outcomes: list[SubQuestionOutcome]) -> bool:
        try:
            return all(
                (getattr(getattr(f, "layout", None), "meta", None) or {}).get("result_hash") == o.contract.result_hash
                for o in outcomes if o.contract is not None for f in o.figures)
        except Exception as exc:  # noqa: BLE001 - a malformed figure must not fail the answer
            _log_exc("chart reconciliation check failed", exc)
            return False

    def _judge(self, user_query: str, subs: list[dict], outcomes: list[SubQuestionOutcome],
               headline: str, narrative: str):
        """One QA-gate call. Never raises: an unavailable gate leaves the verdict to the local checks."""
        draft = {"headline": headline, "narrative": narrative}
        try:
            if self._review_takes_evidence:
                return self.responder.review(user_query, subs, draft, evidence=self._outcomes_payload(outcomes))
            return self.responder.review(user_query, subs, draft)
        except Exception as exc:  # noqa: BLE001
            _log_exc("response validation failed; continuing without the QA gate", exc)
            return None

    def _validate_response(self, user_query: str, subs: list[dict], outcomes: list[SubQuestionOutcome],
                           headline: str, narrative: str) -> dict:
        unanswered = [o.id for o in outcomes if o.status != "ANSWERED"]
        answered_ids = {o.id for o in outcomes}
        missing = [s["id"] for s in subs if s["id"] not in answered_ids]
        verdict: dict[str, Any] = {
            "verdict": "PASS" if not unanswered and not missing else "FAIL",
            "defects": [], "unanswered_ids": unanswered + missing,
            "charts_reconciled": self._charts_reconciled(outcomes),
            "dropped_numeric_claims": sum(len(o.dropped_claims) for o in outcomes),
        }
        if len(subs) < 2:
            return verdict
        judged = self._judge(user_query, subs, outcomes, headline, narrative)
        if judged is not None:
            # Blocking defects first, so the 5-item cut-off can never hide the one that matters.
            ordered = sorted(judged.defects, key=lambda d: d.severity != "blocking")
            verdict["defects"] = [d.model_dump() for d in ordered[:5]]
            verdict["blocking_defects"] = sum(d.severity == "blocking" for d in judged.defects)
            if judged.verdict == "FAIL":
                verdict["verdict"] = "FAIL"
            plan_ids = {s["id"] for s in subs}
            for uid in judged.unanswered_ids:
                if uid not in verdict["unanswered_ids"] and uid in plan_ids:
                    verdict["unanswered_ids"].append(uid)
        return verdict

    @staticmethod
    def _gap_disclosure(unanswered_ids: list, outcomes: list[SubQuestionOutcome]) -> str:
        by_id = {o.id: o for o in outcomes}
        lines = [f"- **{by_id[u].question_text.strip()}** — {by_id[u].failure_reason or 'Flagged by the QA review.'}"
                 for u in unanswered_ids if u in by_id]
        return ("\n\n**Not answered from the available data**\n\n" + "\n".join(lines)) if lines else ""

    # ── output ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_insight_card(outcomes: list[SubQuestionOutcome], headline: str) -> dict:
        """First answered question is the top card; all answered ones go under ``questions``."""
        answered = [o for o in outcomes if o.status == "ANSWERED" and o.insight]
        if not answered:
            return {}
        cards = [{**o.insight, "question": o.question_text,
                  "_sql": o.contract.sql if o.contract is not None else ""} for o in answered]
        lead = dict(cards[0])
        lead["questions"] = cards
        lead["_sql"] = "\n\n".join(f"-- {o.id}: {o.question_text}\n{o.contract.sql}"
                                   for o in answered if o.contract is not None)
        if len(cards) > 1:
            lead["insight"] = headline
            lead["key_finding"] = {"text": headline, "refs": []}
        return lead


__all__ = ["EngineConfig", "AnalyticsEngine", "ResultContract", "SubQuestionOutcome", "PipelineResult",
           "UNRELATED_MESSAGE", "dep_ids"]