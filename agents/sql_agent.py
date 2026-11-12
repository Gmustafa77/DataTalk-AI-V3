"""SQL agent — writes one governed Databricks SELECT per sub-question.

LLM calls per sub-question:
  * 1 × ``sql_generation`` (always, when the sub-question is answerable),
  * 0–1 × ``sql_repair`` (only when the read-only validator rejects the SQL and
    ``ENABLE_SQL_REPAIR`` is on).

Every statement is checked by ``core.sql_guard.validate_read_only`` before it can
run; the agent never executes SQL itself.

Backward compatibility
----------------------
* ``SqlAgent.__init__`` / ``generate`` keep their old signatures; new parameters are
  keyword-only with defaults that reproduce the old behaviour.
* ``SqlGeneration``, ``Measure``, ``SqlOutcome`` keep every old field.
* ``build_system_prompt`` keeps its signature; ``REPAIR_SYSTEM_PROMPT`` is kept as a
  constant, and ``build_repair_system_prompt`` is the context-complete version the
  agent now uses.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

import prompts as P
from core.llm import LLMResponseError, LLMUnavailableError, StructuredLLM
from core.schemas import Grain
from core.sql_guard import SqlCheck

log = logging.getLogger("datatalk.sql_agent")

SqlStatus = Literal["SUPPORTED", "UNRELATED", "MISSING_FIELD", "INVALID_FILTER", "AMBIGUOUS",
                    "UNSUPPORTED_METRIC"]

_OPEN_FENCE = re.compile(r"^\s*```(?:sql)?\s*", re.IGNORECASE)
_CLOSE_FENCE = re.compile(r"\s*```\s*$")


def _clean_sql(sql: str) -> str:
    """Remove markdown fences and trailing semicolons the model sometimes adds.

    Only the *trailing* terminator is stripped; an inner ``;`` (multi-statement) is left
    for the read-only validator to reject.
    """
    text = _CLOSE_FENCE.sub("", _OPEN_FENCE.sub("", sql)).strip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def _norm_name(name: Any) -> str:
    """Spaces/underscores/case-insensitive key so 'Provider Name' == 'Provider_Name'."""
    return re.sub(r"[\s_]+", "", str(name)).lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Response contract (shared by generation and repair)
# ═══════════════════════════════════════════════════════════════════════════════
class Measure(BaseModel):
    column: str
    type: str = ""
    numerator: str | None = None
    denominator: str | None = None


class SqlGeneration(BaseModel):
    answerable: bool
    status: SqlStatus = "SUPPORTED"
    reason: str = ""
    sql: str = ""
    intent: str = ""
    referenced_columns: list[str] = Field(default_factory=list)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    grain: Grain = Grain.NONE
    dimensions: list[str] = Field(default_factory=list)
    measures: list[Measure | str] = Field(default_factory=list)
    expected_row_scale: str = ""
    is_aggregate: bool = False
    self_check: dict[str, Any] = Field(default_factory=dict)
    repair_applied: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        if v is None:
            return "SUPPORTED"
        return v.strip().upper().replace(" ", "_").replace("-", "_") if isinstance(v, str) else v

    @field_validator("grain", mode="before")
    @classmethod
    def _grain(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else (v or "none")

    @field_validator("filters_applied", "self_check", mode="before")
    @classmethod
    def _dicts(cls, v: Any) -> Any:
        return {} if v is None else v

    @field_validator("referenced_columns", "dimensions", "measures", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> Any:
        return [] if v is None else v

    @field_validator("reason", "intent", "expected_row_scale", "repair_applied", mode="before")
    @classmethod
    def _strings(cls, v: Any) -> Any:
        return "" if v is None else v

    @field_validator("sql", mode="before")
    @classmethod
    def _sql(cls, v: Any) -> Any:
        if v is None:
            return ""
        return _clean_sql(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _consistent(self) -> SqlGeneration:
        if self.answerable:
            if not self.sql.strip():
                raise ValueError("answerable=true requires a non-empty sql statement")
            if self.status != "SUPPORTED":
                self.status = "SUPPORTED"
        else:
            if self.status == "SUPPORTED":
                self.status = "AMBIGUOUS"  # a failure object must carry a failure status
            self.sql = ""  # a failure object never carries SQL
        return self


def repair_scope_violations(original: SqlGeneration, repaired: SqlGeneration) -> list[str]:
    """Return reasons a repair changed the question's scope (empty list = scope preserved).

    A repair may fix columns, functions and syntax. It must not drop a filter, change the
    time grain, or change the grouping dimensions. Filter *values* may change (a repair can
    legitimately correct casing or a code), so only the filter keys are compared.
    """
    problems: list[str] = []
    dropped = {_norm_name(k) for k in original.filters_applied} - {_norm_name(k) for k in repaired.filters_applied}
    if dropped:
        problems.append("dropped filters: " + ", ".join(sorted(dropped)))
    if original.grain != repaired.grain:
        problems.append(f"grain changed: {original.grain} -> {repaired.grain}")
    if {_norm_name(d) for d in original.dimensions} != {_norm_name(d) for d in repaired.dimensions}:
        problems.append("dimensions changed")
    return problems


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════
_SQL_RESPONSE_SCHEMA = """RESPONSE SCHEMA
{
  "answerable": true,
  "status": "SUPPORTED",
  "reason": "",
  "sql": "SELECT ...",
  "intent": "one sentence describing precisely what this query returns",
  "referenced_columns": ["Case Status", "Determination"],
  "filters_applied": {"Received Date": {"from": "2026-04-01", "to": "2026-09-30"}},
  "grain": "month",
  "dimensions": ["Current Unit Name"],
  "measures": [
    {"column": "Overturn Rate", "type": "percentage", "numerator": "Overturned Cases", "denominator": "Closed Cases"}
  ],
  "expected_row_scale": "one row per unit per month",
  "is_aggregate": true,
  "self_check": {"columns_valid": true, "denominators_valid": true, "division_guarded": true, "shape_matches_intent": true, "order_deterministic": true}
}"""

_SQL_FAILURE_OBJECT = """FAILURE OBJECT
{"answerable": false,
  "status": "UNRELATED" | "MISSING_FIELD" | "INVALID_FILTER" | "AMBIGUOUS" | "UNSUPPORTED_METRIC",
  "reason": "one plain-English sentence a business user can act on",
  "sql": "", "intent": "", "referenced_columns": [], "filters_applied": {},
  "grain": "none", "dimensions": [], "measures": [], "expected_row_scale": "",
  "is_aggregate": false, "self_check": {}}"""

_DEPENDENT_RULES = """DEPENDENT-QUESTION RULES
- `upstream_results` may contain verified rows from a prerequisite sub-question.
- When the current question says "its", "those", "that sponsor", "that provider",
  or otherwise depends on a winner identified upstream, resolve the reference from
  `upstream_results`. Do not ask the user to repeat it.
- Use only literal values present in upstream_results; never invent a value.
- If the required upstream task failed or has no rows, return AMBIGUOUS with a clear reason."""

_DATE_RULES = """DATE RESOLUTION
- `as_of_date` in the input is today's date. Resolve every relative period ("last 30
  days", "this month", "last quarter", "YTD", "this year") to literal ISO dates computed
  from `as_of_date`, and report them in `filters_applied`.
- Certified metric expressions that use current_date() stay exactly as written.
- If the user gives no resolvable period, apply no date filter unless the question
  needs one. Never invent a period."""

_SELF_CHECK = """SELF-CHECK BEFORE RETURNING (perform silently, then report in `self_check`)
 1. Does every column in my SQL appear verbatim in the raw column list?
 2. Is every rate's denominator correct per the denominator discipline?
 3. Is every division guarded against zero/NULL?
 4. Does the result shape match the requested analysis_type and grain?
 5. Is row order deterministic?
 6. Have I avoided every forbidden pattern?
If any check fails, fix the SQL and re-run the checks before responding."""


def _sql_rules(fq_table: str, max_row_query_limit: int) -> str:
    """Construction / shape / forbidden rules shared by generation and repair."""
    return f"""SQL CONSTRUCTION RULES
- Certified metric expressions are used verbatim and take precedence over any generic
  rule below. Metric names such as `total_appeals` are logical names, never output aliases.
- Dialect: Databricks SQL (Spark SQL). Use only Databricks-supported functions.
- Exactly one statement. No semicolon. No SQL comments (`--`, `/* */`).
- Read-only. INSERT, UPDATE, DELETE, MERGE, DROP, CREATE, ALTER, TRUNCATE, COPY,
  GRANT, REVOKE, OPTIMIZE, VACUUM, CALL, EXECUTE, USE, and SET are forbidden.
- `SELECT *` is forbidden. Project named columns only.
- Only {fq_table} may appear in FROM/JOIN. Self-joins on this table are permitted
  only when unavoidable; prefer window functions or conditional aggregation.
- Wrap every column reference and every alias in backticks.
- Alias every computed column with a clear, human-readable label without underscores
  that a chart axis can use directly, e.g. `Overturn Rate`, `Closed Cases`, `Month`.
- NULL SAFETY: guard every division with a CASE or NULLIF so a zero denominator
  returns NULL, never a divide-by-zero and never a silent 0.
- CAST explicitly when mixing integer and decimal arithmetic. Percentages must be
  computed in double precision and ROUND(...) to 2 decimals.
- Emit the denominator of every rate as its own column so downstream validation can
  recompute the rate independently.
- Deterministic ordering: every non-aggregate result, and every query that uses LIMIT,
  must have an ORDER BY that fully determines row order. Ties must be broken by a
  unique-ish column.
- Time grain: when the sub-question has a grain, emit the period as a STRING column
  formatted 'yyyy-MM' (month), 'yyyy-Qq' (quarter), or 'yyyy-MM-dd' (day), and
  ORDER BY it ascending. Never return a raw timestamp for a trend axis.
- Filter values must come from `available_values` exactly as listed. For free-text
  filters not covered by a certified metric expression, compare upper(trim(col)) with
  upper(trim('<value>')). If a requested value is not in `available_values`, return
  INVALID_FILTER.

RESULT SHAPE RULES (these directly control chart correctness)
- Aggregation queries: no LIMIT required, but a ranking must use an explicit
  LIMIT N matching the user's N (default 10 if the user said "top" without N).
- Row-level queries: LIMIT {max_row_query_limit} is mandatory.
- Never return more than 25 grouped rows for a breakdown unless the user explicitly
  asked for the full list — rank and LIMIT instead. Oversized breakdowns produce
  unreadable charts.
- One row per (dimension combination, period). Never emit duplicate keys; the chart
  layer treats duplicate keys as a data-integrity failure and will reject the query.

FORBIDDEN PATTERNS
- Referencing any Python-only derived field.
- Hardcoding a count, rate, date, or threshold that should be computed.
- Using an industry benchmark or target that was not supplied.
- Returning both a raw row set and an aggregate in the same statement.
- Window functions over an unbounded frame without an explicit PARTITION BY when
  the sub-question is scoped to a dimension."""


def _trust_rules() -> str:
    # getattr keeps this module working with the previous prompts.py.
    return getattr(P, "UNTRUSTED_INPUT_RULES", "")


def build_system_prompt(
    fq_table: str,
    raw_columns: Iterable[str],
    derived_columns: Iterable[str],
    sla_days: int,
    escalation_level: int,
    max_row_query_limit: int,
    column_types: dict | None = None,
) -> str:
    return f"""You generate exactly ONE safe, read-only Databricks SQL SELECT
statement for ONE sub-question of a healthcare Appeals analytics request. You are
a compiler, not an analyst. You produce SQL and metadata — never narrative.

{P.schema_block(fq_table, raw_columns, derived_columns, column_types)}

{P.metric_dictionary(sla_days, escalation_level)}

{_DEPENDENT_RULES}

{_sql_rules(fq_table, max_row_query_limit)}

{_DATE_RULES}

{_SELF_CHECK}

{_trust_rules()}

{P.ANTI_HALLUCINATION}

{P.JSON_DISCIPLINE}

{_SQL_RESPONSE_SCHEMA}

{_SQL_FAILURE_OBJECT}"""


# Kept as a constant for backward compatibility (tests / external imports). The agent
# uses build_repair_system_prompt(), which adds the context this text refers to.
REPAIR_SYSTEM_PROMPT = f"""You repair ONE rejected Databricks SQL statement for a
healthcare Appeals analytics system. You will receive the original sub-question,
the rejected SQL, and the exact validator or engine errors.

REPAIR RULES
- Fix ONLY what the errors identify. Do not restructure, re-scope, or "improve"
  a query that was otherwise correct.
- Do not broaden filters, drop a WHERE clause, or remove a GROUP BY to make an
  error disappear. Silently changing the question is worse than failing.
- If the error is an unknown column, map it to the correct raw column from the
  data contract. If no correct column exists, do NOT guess — return the failure
  object with status MISSING_FIELD.
- If the error is a Python-only derived field, recreate the equivalent logic from
  raw columns using the certified metric dictionary.
- If the error indicates the result was too large, add the correct aggregation or
  LIMIT rather than truncating meaning.
- You get exactly one repair attempt. If you are not confident the repair is
  correct, return the failure object.

{P.ANTI_HALLUCINATION}

{P.JSON_DISCIPLINE}

Return the SAME schema as the SQL generator, plus:
  "repair_applied": "one sentence describing exactly what you changed"
"""


def build_repair_system_prompt(
    fq_table: str,
    raw_columns: Iterable[str],
    derived_columns: Iterable[str],
    sla_days: int,
    escalation_level: int,
    max_row_query_limit: int,
    column_types: dict | None = None,
) -> str:
    """Repair prompt WITH the data contract, metric dictionary, SQL rules and schema.

    The bare ``REPAIR_SYSTEM_PROMPT`` tells the model to "map to the correct raw column from
    the data contract" but never supplies that contract, which invites guessed columns.
    """
    return f"""{REPAIR_SYSTEM_PROMPT}
{_trust_rules()}

REFERENCE — the data contract the repaired SQL must satisfy
{P.schema_block(fq_table, raw_columns, derived_columns, column_types)}

{P.metric_dictionary(sla_days, escalation_level)}

REFERENCE — the SQL rules the repaired statement must still satisfy
{_sql_rules(fq_table, max_row_query_limit)}

{_DATE_RULES}

REFERENCE — the response schema that "the SAME schema" refers to (add "repair_applied")
{_SQL_RESPONSE_SCHEMA}

{_SQL_FAILURE_OBJECT}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SqlOutcome:
    """Result of generate → validate → (repair → validate)."""

    generation: SqlGeneration | None
    check: SqlCheck | None
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.generation is not None and self.check is not None and self.check.valid


class SqlAgent:
    def __init__(self, llm: StructuredLLM, validate: Callable[[str], SqlCheck], *, fq_table: str,
                 raw_columns: Iterable[str], derived_columns: Iterable[str], sla_days: int,
                 escalation_level: int, max_row_query_limit: int, enable_repair: bool = True,
                 column_types: dict | None = None, strict_repair_scope: bool = True) -> None:
        self.llm = llm
        self.validate = validate
        self.enable_repair = enable_repair
        self.strict_repair_scope = strict_repair_scope
        # Materialise once: a generator would be exhausted by the first prompt build.
        raw = list(raw_columns)
        derived = list(derived_columns)
        self._raw_columns = frozenset(raw)
        self.system = build_system_prompt(fq_table, raw, derived, sla_days, escalation_level,
                                          max_row_query_limit, column_types)
        self.repair_system = build_repair_system_prompt(fq_table, raw, derived, sla_days,
                                                        escalation_level, max_row_query_limit, column_types)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _safe_validate(self, sql: str) -> SqlCheck | None:
        """Run the read-only validator; a crashing validator must never crash the request."""
        try:
            return self.validate(sql)
        except Exception as exc:  # noqa: BLE001 - validator is external code
            log.error("sql validator raised %s", type(exc).__name__)
            return None

    def _record_self_check(self, gen: SqlGeneration, diag: dict[str, Any]) -> None:
        failed = sorted(k for k, v in gen.self_check.items() if v is False)
        if failed:
            diag["self_check_failed"] = failed
            log.warning("model self-check reported failures: %s", ", ".join(failed))
        unknown = [c for c in gen.referenced_columns if c not in self._raw_columns]
        if unknown:
            diag["unknown_referenced_columns"] = unknown

    def _try_repair(self, sub_question: dict, rejected_sql: str, errors: Iterable[str], *,
                    as_of_date: str, available_values: dict, upstream_results: list | None,
                    original: SqlGeneration | None, diag: dict[str, Any]
                    ) -> tuple[SqlGeneration, SqlCheck] | None:
        """One repair attempt. Returns (generation, check) only when the repair is valid AND in scope."""
        diag["repair_attempted"] = True
        payload = P.user_payload(sub_question=sub_question, rejected_sql=rejected_sql, errors=list(errors),
                                 as_of_date=as_of_date, available_values=available_values,
                                 upstream_results=upstream_results or [])
        try:
            repaired = self.llm.structured("sql_repair", self.repair_system, payload, SqlGeneration)
        except (LLMResponseError, LLMUnavailableError) as exc:
            log.warning("sql repair failed: %s", type(exc).__name__)
            diag["repair_error"] = type(exc).__name__
            return None
        if not repaired.answerable:
            diag["repair_status"] = repaired.status
            diag["repair_reason"] = repaired.reason
            return None
        if original is not None and self.strict_repair_scope:
            problems = repair_scope_violations(original, repaired)
            if problems:
                diag["repair_rejected_scope"] = problems
                log.info("sql repair rejected, scope changed: %s", "; ".join(problems))
                return None
        check = self._safe_validate(repaired.sql)
        if check is None or not check.valid:
            diag["repair_validator_errors"] = list(check.errors) if check is not None else ["validator_error"]
            return None
        diag["repair_applied"] = repaired.repair_applied
        return repaired, check

    # ── public API ─────────────────────────────────────────────────────────────
    def generate(self, sub_question: dict, *, as_of_date: str, available_values: dict,
                 upstream_results: list | None = None) -> SqlOutcome:
        diag: dict[str, Any] = {}
        payload = P.user_payload(sub_question=sub_question, as_of_date=as_of_date,
                                 available_values=available_values,
                                 upstream_results=upstream_results or [])
        try:
            gen = self.llm.structured("sql_generation", self.system, payload, SqlGeneration)
        except (LLMResponseError, LLMUnavailableError) as exc:
            log.warning("sql generation failed: %s", type(exc).__name__)
            return SqlOutcome(None, None, "The AI service could not generate a query for this question. "
                                          "Please retry.", {"error_type": type(exc).__name__})
        diag["self_check"] = gen.self_check
        if not gen.answerable:
            return SqlOutcome(gen, None, gen.reason or "No supported query could be generated.", diag)
        self._record_self_check(gen, diag)

        check = self._safe_validate(gen.sql)
        if check is None:
            diag["validator_error"] = True
            return SqlOutcome(gen, None, "The query safety check could not be completed, so the query was "
                                         "not run. Please retry.", diag)

        if not check.valid and self.enable_repair:
            fixed = self._try_repair(sub_question, gen.sql, check.errors, as_of_date=as_of_date,
                                     available_values=available_values, upstream_results=upstream_results,
                                     original=gen, diag=diag)
            if fixed is not None:
                gen, check = fixed

        if not check.valid:
            diag["validator_errors"] = list(check.errors)
            log.info("sql blocked by the read-only policy: %s", "; ".join(check.errors))
            return SqlOutcome(gen, check, "The generated query did not pass the read-only safety checks, so it "
                                          "was not run. Try rephrasing with the business terms used in the data "
                                          "(units, sponsors, providers, statuses, dates).", diag)
        return SqlOutcome(gen, check, "", diag)

    def repair(self, sub_question: dict, rejected_sql: str, errors: Iterable[str], *, as_of_date: str,
               available_values: dict, upstream_results: list | None = None,
               original: SqlGeneration | None = None) -> SqlOutcome:
        """Optional: repair SQL that passed the validator but failed in the engine.

        Same single-attempt, scope-preserving, re-validated repair used inside ``generate``.
        """
        diag: dict[str, Any] = {"repair_source": "engine_error"}
        if not self.enable_repair:
            return SqlOutcome(original, None, "Automatic repair is disabled.", diag)
        fixed = self._try_repair(sub_question, rejected_sql, errors, as_of_date=as_of_date,
                                 available_values=available_values, upstream_results=upstream_results,
                                 original=original, diag=diag)
        if fixed is None:
            return SqlOutcome(original, None, "The query could not be safely repaired. Try rephrasing with the "
                                              "business terms used in the data.", diag)
        return SqlOutcome(fixed[0], fixed[1], "", diag)