"""Planner agent — splits one user message into executable sub-questions.

LLM calls: exactly one logical call (purpose ``planner``) per analytics request.
If that call fails or returns an unusable plan, a deterministic rule-based splitter
creates the sub-questions instead, so a question is never silently dropped.

Input : user question, ``as_of_date``, grounding values, last 3 conversation turns.
Output: ``(overall_status, [sub-question dict])`` for the engine.

Guarantees after ``plan()`` (all enforced in code, not left to the LLM):
  * ids are unique; ``depends_on`` is a list of known ids with no self-loops or cycles
  * ``answerable`` is False whenever ``status`` says the question cannot be answered
  * a question the LLM missed is recovered from the user's message
  * questions beyond ``max_sub_questions`` are reported (status OVER_LIMIT), never dropped silently
  * ``dimensions`` use the exact column spellings; reversed date ranges are swapped
  * ``overall_status`` is derived from the sub-questions, not trusted from the LLM
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

import prompts as P
from core.llm import LLMResponseError, LLMUnavailableError, StructuredLLM
from core.schemas import Grain

log = logging.getLogger("datatalk.planner")

ANALYSIS_TYPES = ("metric", "breakdown", "trend", "comparison", "ranking", "distribution",
                  "drilldown", "forecast", "anomaly", "root_cause", "overview")
CHART_HINTS = ("line", "area", "bar", "stacked_bar", "grouped_bar", "pie", "donut", "scatter",
               "heatmap", "kpi", "none")
AnalysisType = Literal["metric", "breakdown", "trend", "comparison", "ranking", "distribution",
                       "drilldown", "forecast", "anomaly", "root_cause", "overview"]
ChartHint = Literal["line", "area", "bar", "stacked_bar", "grouped_bar", "pie", "donut", "scatter",
                    "heatmap", "kpi", "none"]
PlanItemStatus = Literal["SUPPORTED", "PARTIAL", "UNRELATED", "AMBIGUOUS", "INVALID_FILTER",
                         "MISSING_FIELD", "OVER_LIMIT"]
# Statuses that mean "do not run SQL for this one".
BLOCKING_STATUSES = frozenset({"UNRELATED", "AMBIGUOUS", "INVALID_FILTER", "MISSING_FIELD", "OVER_LIMIT"})

_MIN_WORDS = 4           # fragments shorter than this are merged into a neighbour, never dropped
_COVERAGE = 0.6          # share of a fragment's words a sub-question must contain to "cover" it
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(v: Any) -> Any:
    return v.strip().lower().replace(" ", "_") if isinstance(v, str) else v


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    items = list(v) if isinstance(v, (list, tuple, set)) else [v]
    out: list[str] = []
    for item in items:
        s = str(item).strip()
        if s and s.lower() not in ("null", "none") and s not in out:
            out.append(s)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Response contract
# ═══════════════════════════════════════════════════════════════════════════════
class SubQuestion(BaseModel):
    id: str = Field(min_length=1, max_length=8, description="q1, q2, ... in execution order")
    question_text: str = Field(min_length=1, description="Verbatim slice of the user's message")
    resolved_text: str = Field(default="", description="Self-contained rewrite, only for follow-ups")
    answerable: bool = True
    status: PlanItemStatus = "SUPPORTED"
    reason: str = ""
    analysis_type: AnalysisType = "metric"
    grain: Grain = Grain.NONE
    dimensions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    dependency_note: str = ""
    expects_chart: bool = True
    chart_hint: ChartHint = "none"

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> Any:
        return str(v).strip()[:8] if v is not None else v

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    # One unknown enum value must not throw away the whole LLM plan, so these coerce to a safe default.
    @field_validator("analysis_type", mode="before")
    @classmethod
    def _analysis(cls, v: Any) -> Any:
        v = _norm(v)
        return v if v in ANALYSIS_TYPES else "metric"

    @field_validator("chart_hint", mode="before")
    @classmethod
    def _hint(cls, v: Any) -> Any:
        v = _norm(v)
        return v if v in CHART_HINTS else "none"

    @field_validator("grain", mode="before")
    @classmethod
    def _grain(cls, v: Any) -> Any:
        if isinstance(v, Grain):
            return v
        try:
            return Grain(_norm(v) or "none")
        except ValueError:
            return Grain.NONE

    @field_validator("dimensions", "metrics", "depends_on", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> Any:
        return _as_str_list(v)

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else {}

    @field_validator("reason", "dependency_note", "resolved_text", mode="before")
    @classmethod
    def _text(cls, v: Any) -> Any:
        return "" if v is None else v

    @model_validator(mode="after")
    def _consistent(self) -> "SubQuestion":
        if self.status in BLOCKING_STATUSES:
            self.answerable = False          # e.g. INVALID_FILTER + answerable=true would run bad SQL
        if not self.answerable and not self.reason.strip():
            self.reason = "This question cannot be answered from the Appeals dataset."
        return self


class QueryPlan(BaseModel):
    overall_status: Literal["SUPPORTED", "PARTIAL", "UNRELATED"]
    question_count: int = Field(default=0, ge=0)
    # Capped by the agent (with a visible OVER_LIMIT notice) rather than rejected here.
    sub_questions: list[SubQuestion] = Field(default_factory=list, max_length=50)

    @field_validator("overall_status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════════════════════
def build_system_prompt(
    fq_table: str,
    raw_columns: Iterable[str],
    derived_columns: Iterable[str],
    sla_days: int,
    escalation_level: int,
) -> str:
    return f"""You are the Query Understanding and Planning stage of a regulated
healthcare Appeals analytics system. You DO NOT answer questions and you DO NOT
write SQL. You only decompose the user's input into an executable task plan.

{P.schema_block(fq_table, raw_columns, derived_columns)}

{P.metric_dictionary(sla_days, escalation_level)}

DECOMPOSITION RULES
- A single user message may contain MULTIPLE questions. You must find every one of
  them. Missing a question is the single worst failure mode of this system.
- Treat each of the following as a separate question:
    * sentences separated by '?', ';', ' and also ', ' also ', ' plus ',
      ' as well as ', numbered or bulleted lists
    * a comparison that spans two populations ("X vs Y") when the user also asks
      for a driver, reason, or trend
    * a request for a number PLUS a request for an explanation, trend, breakdown,
      forecast, or recommendation
- Do NOT split a single coherent analytical request into artificial fragments.
  "overturn rate by provider for the last 3 months" is ONE question.
  A short modifier such as "Show the top 5." belongs to the question before it.
- Preserve the user's original wording in `question_text` for every sub-question.
- Order sub-questions by execution order. If a sub-question needs inputs from
  the result of one or more earlier sub-questions (for example "then break those
  providers down by specialty"), set `depends_on` to the list of those ids and
  explain the dependency in `dependency_note`. `depends_on` may only reference
  EARLIER ids, never itself and never a later question.
- Independent sub-questions MUST have depends_on = [] so they can be executed in
  parallel.
- FOLLOW-UPS: when a sub-question only makes sense with `recent_turns` (for example
  "what about last month?"), keep `question_text` verbatim, fill `resolved_text` with
  a self-contained version that carries over the metric, dimension and filters from
  the earlier turn, and copy those into `metrics`, `dimensions` and `filters`.
  Leave `resolved_text` empty when the sub-question stands on its own.

PER SUB-QUESTION CLASSIFICATION
analysis_type must be exactly one of:
  metric        - a single scalar or small set of scalars
  breakdown     - one metric grouped by one or more dimensions
  trend         - a metric over a time grain
  comparison    - the same metric across two or more named populations/periods
  ranking       - top/bottom N by a metric
  distribution  - share/mix/composition of a whole
  drilldown     - row-level records matching a filter
  forecast      - projection beyond the observed period
  anomaly       - outlier or deviation detection
  root_cause    - decomposition of a change into contributing factors
  overview      - a compact portfolio summary using the certified overview metric set

status must be exactly one of:
  SUPPORTED       - fully answerable from the data contract
  PARTIAL         - answerable, but part of the request cannot be satisfied (say which in `reason`)
  AMBIGUOUS       - genuinely undecidable; put the clarifying question in `reason`
  INVALID_FILTER  - a filter value is not present in `available_values`
  MISSING_FIELD   - the data contract has no column or metric that can answer it
  UNRELATED       - not about the Appeals dataset
`answerable` MUST be false for AMBIGUOUS, INVALID_FILTER, MISSING_FIELD and UNRELATED,
and true for SUPPORTED and PARTIAL.

grain: the time grain requested, one of day, week, month, quarter, year, none.
dimensions: dimension columns the answer must be grouped by, using EXACT raw
  column spellings from the data contract. Empty list if none.
metrics: metric keys from the certified metric dictionary. Never free-text.
filters: an object of resolved filters. Resolve relative dates against
  `as_of_date` supplied in the user payload and emit ISO dates (YYYY-MM-DD), never
  phrases like "last month". If a filter value is not present in `available_values`,
  do not guess — mark the sub-question unanswerable with status INVALID_FILTER.
expects_chart: true when a visual materially helps (trend, breakdown, comparison,
  ranking, distribution). false for a single scalar or a row-level drilldown.
chart_hint: one of line, area, bar, stacked_bar, grouped_bar, pie, donut, scatter,
  heatmap, kpi, none. This is ADVISORY ONLY — the rendering layer may override it.

AMBIGUITY POLICY
- Resolve ambiguity in favor of the certified metric dictionary. Do not ask the
  user a clarifying question unless the request is genuinely undecidable.
- If undecidable, emit the sub-question with answerable=false and status AMBIGUOUS,
  and put the single most useful clarifying question in `reason`.
- Partial plans are allowed and expected: some sub-questions may be answerable and
  others not. Never drop an unanswerable sub-question — report it.

{P.ANTI_HALLUCINATION}

{P.JSON_DISCIPLINE}

RESPONSE SCHEMA
{{
  "overall_status": "SUPPORTED" | "PARTIAL" | "UNRELATED",
  "question_count": 2,
  "sub_questions": [
    {{
      "id": "q1",
      "question_text": "verbatim slice of the user's message",
      "resolved_text": "",
      "answerable": true,
      "status": "SUPPORTED",
      "reason": "",
      "analysis_type": "trend",
      "grain": "month",
      "dimensions": ["Current Unit Name"],
      "metrics": ["overturn_rate"],
      "filters": {{"Received Date": {{"from": "2026-04-01", "to": "2026-09-30"}}}},
      "depends_on": [],
      "dependency_note": "",
      "expects_chart": true,
      "chart_hint": "line"
    }}
  ]
}}

FAILURE OBJECT (use when the whole message is off-domain)
{{"overall_status": "UNRELATED", "question_count": 0, "sub_questions": []}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic splitter
# ═══════════════════════════════════════════════════════════════════════════════
_SPLIT_RE = re.compile(
    r"(?:(?<=\?)\s+)"
    r"|\s*;\s*"
    r"|\s+(?:and\s+also|also\s+tell\s+me|also\s+show|as\s+well\s+as|and\s+then|plus\s+also)\s+"
    r"|\s*\n\s*(?=\d+[\.\)]\s)"
    r"|\s*\n\s*(?=[-*•]\s)",
    flags=re.I,
)


def rule_based_split(text: str, max_parts: int | None = None) -> list[str]:
    """Deterministic splitter on '?', ';', 'and also', numbered and bulleted lists.

    Fragments with fewer than ``_MIN_WORDS`` words ("Why?", "Top 5.") are merged into the
    neighbouring fragment instead of being dropped or turned into a meaningless question.
    ``max_parts=None`` returns every fragment; the caller decides how to report overflow.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    pieces = [re.sub(r"^\d+[\.\)]\s*", "", p.strip(" -*•\t")).strip() for p in _SPLIT_RE.split(raw) if p]

    parts: list[str] = []
    carry = ""
    for piece in pieces:
        if not piece:
            continue
        if carry:
            piece, carry = f"{carry} {piece}", ""
        if len(piece.split()) < _MIN_WORDS:
            if parts:
                parts[-1] = f"{parts[-1]} {piece}"
            else:
                carry = piece
        else:
            parts.append(piece)
    if carry:
        parts.append(carry)

    out, seen = [], set()
    for part in parts:
        if part.lower() not in seen:
            seen.add(part.lower())
            out.append(part)
    out = out or [raw]
    return out[:max_parts] if max_parts else out


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 2}


def _make_sub(sub_id: str, text: str, **extra: Any) -> dict:
    return SubQuestion(id=sub_id, question_text=text, chart_hint="none", **extra).model_dump(mode="json")


def _iso(v: Any) -> date | None:
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════
class PlannerAgent:
    """Turns a user message into an ordered list of sub-question dicts."""

    purpose = "planner"

    def __init__(self, llm: StructuredLLM, *, fq_table: str, raw_columns: Iterable[str],
                 derived_columns: Iterable[str], sla_days: int, escalation_level: int,
                 max_sub_questions: int) -> None:
        raw_columns, derived_columns = list(raw_columns), list(derived_columns)  # may be generators
        self.llm = llm
        self.max_sub_questions = max(1, max_sub_questions)
        self._columns = {str(c).lower(): str(c) for c in [*raw_columns, *derived_columns]}
        self.system = build_system_prompt(fq_table, raw_columns, derived_columns, sla_days, escalation_level)

    # ── public ───────────────────────────────────────────────────────────────────
    def plan(self, question: str, *, as_of_date: str, available_values: dict,
             conversation_context: list[str] | None = None) -> tuple[str, list[dict]]:
        """Return ``(overall_status, sub_questions)``.

        ``overall_status`` is ``UNRELATED`` (no sub-questions) when the planner judged the
        whole message off-domain, ``FALLBACK`` when the rule-based splitter was used,
        otherwise ``SUPPORTED`` or ``PARTIAL`` derived from the sub-questions.
        """
        payload = P.user_payload(user_question=question, as_of_date=as_of_date,
                                 available_values=available_values,
                                 recent_turns=(conversation_context or [])[-3:])
        try:
            plan = self.llm.structured(self.purpose, self.system, payload, QueryPlan)
        except (LLMResponseError, LLMUnavailableError) as exc:
            log.warning("planner unavailable, using rule-based split: %s", type(exc).__name__)
            return self._fallback(question)
        except Exception:  # a planner bug must never take the whole request down
            log.exception("planner crashed, using rule-based split")
            return self._fallback(question)

        subs = [s.model_dump(mode="json") for s in plan.sub_questions]
        if not subs:
            return ("UNRELATED", []) if plan.overall_status == "UNRELATED" else self._fallback(question)

        subs = self._unique_ids(subs)
        subs = self._recover_missed(subs, question)
        subs = self._finalise(subs)
        return self._overall(plan.overall_status, subs), subs

    # ── pipeline steps ───────────────────────────────────────────────────────────
    def _fallback(self, question: str) -> tuple[str, list[dict]]:
        subs = [_make_sub(f"q{i + 1}", f) for i, f in enumerate(rule_based_split(question))]
        return "FALLBACK", self._finalise(subs)

    def _finalise(self, subs: list[dict]) -> list[dict]:
        subs = self._fix_dependencies(subs)
        subs = self._cap(subs)
        for s in subs:
            self._sanitise(s)
        return subs

    @staticmethod
    def _unique_ids(subs: list[dict]) -> list[dict]:
        """Renumber q1..qn when the LLM repeats an id, keeping dependencies that were unambiguous."""
        counts = Counter(s["id"] for s in subs)
        if all(c == 1 for c in counts.values()):
            return subs
        mapping = {s["id"]: f"q{i + 1}" for i, s in enumerate(subs) if counts[s["id"]] == 1}
        for i, s in enumerate(subs):
            s["id"] = f"q{i + 1}"
        for s in subs:
            s["depends_on"] = [mapping[d] for d in s.get("depends_on", []) if d in mapping]
        return subs

    def _recover_missed(self, subs: list[dict], question: str) -> list[dict]:
        """Add any part of the user's message that no sub-question covers."""
        fragments = rule_based_split(question)
        if len(fragments) < 2:
            return subs
        if len(subs) == 1:
            # One merged question: trust the splitter, unless the planner flagged it as unanswerable
            # (then the user already gets a reason, and the splitter would erase it).
            if subs[0].get("answerable", True):
                log.info("planner under-split; recovering %d fragments", len(fragments))
                return [_make_sub(f"q{i + 1}", f) for i, f in enumerate(fragments)]
            return subs

        covered = [_words(s["question_text"]) for s in subs]
        used = {s["id"] for s in subs}
        n = len(subs)
        for fragment in fragments:
            words = _words(fragment)
            if not words or any(len(words & c) / len(words) >= _COVERAGE for c in covered):
                continue
            n += 1
            while f"q{n}" in used:
                n += 1
            used.add(f"q{n}")
            covered.append(words)
            subs.append(_make_sub(f"q{n}", fragment))
            log.warning("planner missed a question; recovered %s", f"q{n}")
        return subs

    @staticmethod
    def _fix_dependencies(subs: list[dict]) -> list[dict]:
        """Keep only edges to known ids that do not create a cycle (the later edge of a cycle is dropped)."""
        ids = {s["id"] for s in subs}
        graph: dict[str, list[str]] = {}

        def reaches(start: str, target: str) -> bool:
            stack, seen = [start], set()
            while stack:
                node = stack.pop()
                if node == target:
                    return True
                if node not in seen:
                    seen.add(node)
                    stack.extend(graph.get(node, []))
            return False

        for s in subs:
            kept: list[str] = []
            for d in s.get("depends_on") or []:
                if d == s["id"] or d not in ids:
                    log.warning("dropping invalid dependency %s -> %s", s["id"], d)
                elif reaches(d, s["id"]):
                    log.warning("dropping dependency %s -> %s (would create a cycle)", s["id"], d)
                elif d not in kept:
                    kept.append(d)
            s["depends_on"] = kept
            graph[s["id"]] = kept
        return subs

    def _cap(self, subs: list[dict]) -> list[dict]:
        """Keep the first ``max_sub_questions``; report the rest instead of dropping them silently."""
        limit = self.max_sub_questions
        if len(subs) <= limit:
            return subs
        kept, dropped = subs[:limit], subs[limit:]
        dropped_ids = {s["id"] for s in dropped}
        for s in kept:
            if dropped_ids & set(s["depends_on"]):
                s["depends_on"] = [d for d in s["depends_on"] if d not in dropped_ids]
                s.update(answerable=False, status="OVER_LIMIT",
                         reason="This question depends on a question beyond the per-request limit.")
        text = " | ".join(d["question_text"].strip() for d in dropped)
        used = {s["id"] for s in kept}
        n = limit + 1
        while f"q{n}" in used:
            n += 1
        kept.append(_make_sub(f"q{n}", text[:400], answerable=False, status="OVER_LIMIT",
                              reason=f"Only the first {limit} questions are analysed per request. "
                                     "Please ask the remaining question(s) separately."))
        log.warning("plan had %d sub-questions; %d reported as OVER_LIMIT", len(subs), len(dropped))
        return kept

    def _sanitise(self, s: dict) -> None:
        """Exact column spellings for dimensions; swap reversed date ranges."""
        if self._columns:
            fixed = []
            for d in s.get("dimensions", []):
                exact = self._columns.get(d.lower())
                if exact:
                    fixed.append(exact)
                else:
                    log.warning("%s: dropping unknown dimension %r", s["id"], d)
            s["dimensions"] = fixed
        for key, val in list(s.get("filters", {}).items()):
            if isinstance(val, dict) and "from" in val and "to" in val:
                a, b = _iso(val["from"]), _iso(val["to"])
                if a and b and a > b:
                    s["filters"][key] = {**val, "from": val["to"], "to": val["from"]}

    @staticmethod
    def _overall(llm_status: str, subs: list[dict]) -> str:
        answerable = [s for s in subs if s.get("answerable", True)]
        if len(answerable) == len(subs):
            return "SUPPORTED"
        if answerable:
            return "PARTIAL"
        return "UNRELATED" if llm_status == "UNRELATED" else "PARTIAL"