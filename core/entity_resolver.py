"""Deterministic entity resolution for free-text values in questions (no LLM calls).

Why this exists
---------------
High-cardinality text columns (``Provider_Name``, ``Department``, ``Facility`` ...)
are not part of the grounding values sent to the planner / SQL agent. An exact
value works because the model copies it verbatim into the SQL, but partial names,
aliases, abbreviations, typos or several names in one question ("Sunrise or Sun")
produce ``= 'Sun'`` filters that match nothing, or an ``INVALID_FILTER`` plan.

This module turns the mentions in a question into exact database values, before
any AI call, for EVERY text column of the dataset (nothing is hard-coded):

    dataset ─▶ auto-detect text columns ─▶ one index per column (built once)

    question ─▶ extract mentions ─▶ resolve each mention against every column
              (exact → alias/acronym → token-prefix → substring → fuzzy)
              ─▶ keep the best match level / best columns
              ─▶ grounding block added to ``available_values`` for THIS question

What is dynamic now
-------------------
* Columns: every text-like column is indexed (skips numbers, dates, ID-like
  columns, free-text comment columns). Override with ``ResolverConfig.columns`` /
  ``exclude_columns``.
* "Generic" words (hospital, clinic, department ...) are learned per column from
  the data (words that appear in many values), not from a fixed list.
* Cue words ("providers", "department" ...) come from the column names.
* Values written anywhere in the question (no cue word needed) are found by an
  exact n-gram scan.

Backward compatibility
----------------------
:meth:`EntityResolver.augment` returns the original ``available_values`` object
untouched when nothing is found **or** when every mention is already one exact
full value in one column.

No Streamlit imports.
"""
from __future__ import annotations

import bisect
import difflib
import logging
import re
import threading
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

try:  # optional, much faster fuzzy matching
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover
    fuzz = process = None  # type: ignore[assignment]

log = logging.getLogger("datatalk.entity_resolver")

# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════
# Only words that are meaningless in ANY column. Domain words ("hospital", "clinic")
# are learned per column from the data, or can be added via ResolverConfig.generic_tokens.
UNIVERSAL_GENERIC_TOKENS = frozenset({
    "the", "of", "and", "at", "for", "a", "an", "inc", "llc", "ltd", "pc", "pllc", "pa",
    "corp", "co", "company", "llp", "lp", "plc", "md", "dr",
})
_LEGAL_SUFFIX = frozenset({"inc", "llc", "ltd", "pc", "pllc", "corp", "co", "llp", "lp", "plc", "pa"})

# Words in a question that are never part of an entity mention (question / analytics words).
# Per column, any of these that really occur inside that column's values are un-stopped.
STOP_TOKENS = frozenset({
    "show", "me", "give", "list", "what", "which", "who", "is", "are", "was", "were", "the", "a", "an",
    "average", "avg", "mean", "median", "total", "count", "number", "rate", "rates", "tat", "turnaround",
    "time", "provider", "providers", "appeal", "appeals", "case", "cases", "overturn", "overturned",
    "overturns", "volume", "compare", "comparison", "versus", "vs", "trend", "trends", "monthly",
    "weekly", "daily", "yearly", "by", "per", "each", "all", "top", "bottom", "highest", "lowest",
    "best", "worst", "of", "for", "and", "or", "with", "in", "on", "at", "to", "from", "please",
    "containing", "contains", "contain", "named", "called", "like", "matching", "match", "starting",
    "start", "starts", "beginning", "begins", "name", "names", "their", "its", "them", "those",
    "compliance", "denial", "denials", "upheld", "closed", "open", "pending", "status", "sla",
    "days", "day", "week", "weeks", "month", "months", "quarter", "quarters", "year", "years",
    "last", "past", "previous", "prior", "this", "current", "next", "ytd", "mtd", "qtd", "since",
    "between", "during", "over", "today", "yesterday",
})

# Words that must not become cue words / column hints even if they appear in a column name.
_LABEL_NOISE = frozenset({
    "name", "names", "date", "time", "year", "month", "day", "count", "total", "amount", "rate",
    "avg", "num", "number", "flag", "value", "values", "type", "status", "code", "category", "group",
})

# Match quality, best first.
_TIER = {"exact": 0, "alias": 1, "acronym": 2, "partial": 3, "substring": 4, "fuzzy": 5}

# A mention segment starts after one of these cues (column-name cues are added per dataset) …
_BASE_CUES = [
    r"of", r"for", r"from", r"at", r"about", r"regarding", r"between", r"containing", r"contains",
    r"named", r"called", r"like", r"matching", r"compare", r"starting\s+with",
    r"begin(?:ning|s)?\s+with",
]
# … and stops at one of these (time scope, grouping, another clause).
_STOP_RE = re.compile(
    r"\s+(?:by|per|in|during|over|since|last|past|previous|this|grouped|where|when|which|that|"
    r"broken|split|and\s+(?:show|what|how|which|also|then)|compared)\b"
    r"|[?.;:!()\n]",
    re.I,
)
_SPLIT_RE = re.compile(r"\s*(?:,|/|&|\+|\bor\b|\band\b|\bvs\.?\b|\bversus\b)\s*", re.I)
_QUOTED_RE = re.compile(r"[\"'\u2018\u2019\u201c\u201d`]([^\"'\u2018\u2019\u201c\u201d`]{2,80})[\"'\u2018\u2019\u201c\u201d`]")
_NUMERICISH_RE = re.compile(r"[\d\W_]+")


def normalize(text: Any) -> str:
    """Case-folded, accent-insensitive, punctuation-free form used for matching."""
    t = unicodedata.normalize("NFKD", str(text or ""))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).casefold()
    t = t.replace("&", " and ").replace("'", "")
    t = re.sub(r"[^0-9a-z]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokens(norm: str) -> list[str]:
    return norm.split() if norm else []


def _sql_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _label_words(column: str) -> set[str]:
    """Meaningful words of a column name: ``Provider_Name`` → {"provider"}."""
    parts = re.split(r"[^A-Za-z0-9]+|(?<=[a-z])(?=[A-Z])", str(column))
    return {p.casefold() for p in parts
            if p and len(p) >= 3 and not p.isdigit() and p.casefold() not in _LABEL_NOISE}


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration + results
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ResolverConfig:
    # ── which columns ──
    columns: list | None = None                 # None = auto-detect text columns
    exclude_columns: list = field(default_factory=list)
    exclude_name_pattern: str = r"(^|[_\s])(id|uuid|guid|key|index)$"  # ID-like column names
    min_distinct: int = 2                       # auto-detect: skip columns with fewer values
    max_distinct: int = 200_000                 # auto-detect: skip columns with more values
    max_avg_len: float = 80.0                   # auto-detect: skip free-text (comment) columns
    # ── matching ──
    max_values_per_mention: int = 25            # above this the SQL hint switches to LIKE
    max_values_total: int = 50                  # hard cap on values added per column
    max_columns_per_mention: int = 3            # a mention may match in at most this many columns
    max_mentions: int = 20
    fuzzy_cutoff: float = 88.0                  # 0–100; typo tolerance (rapidfuzz WRatio)
    min_partial_len: int = 3                    # shortest partial term that may match ("Sun")
    scan_exact_values: bool = True              # find exact values anywhere in the question
    ngram_max_tokens: int = 6
    # ── vocabulary ──
    generic_tokens: list = field(default_factory=list)   # extra words never matched alone
    auto_generic_min_distinct: int = 50         # learn generic words only in columns this big
    auto_generic_ratio: float = 0.05            # word in ≥5% of the values …
    auto_generic_min_count: int = 10            # … and in at least this many values
    # {"SRH": "Sunrise Regional Hospital"}  (all columns)
    # or {"Provider_Name": {"SRH": "Sunrise Regional Hospital"}}  (one column)
    aliases: dict = field(default_factory=dict)


@dataclass
class ColumnMatch:
    column: str               # raw SQL column
    match_type: str           # exact | alias | acronym | partial | substring | fuzzy
    values: list[str]         # exact database values
    total_matches: int        # before capping
    like_pattern: str = ""    # set when the mention is partial (used for very wide matches)

    @property
    def truncated(self) -> bool:
        return self.total_matches > len(self.values)


@dataclass
class MentionMatch:
    text: str                       # the mention as written by the user
    candidates: list[ColumnMatch]   # columns it matched in (best match level only)

    @property
    def match_type(self) -> str:
        return self.candidates[0].match_type if self.candidates else ""


# ═══════════════════════════════════════════════════════════════════════════════
# One column
# ═══════════════════════════════════════════════════════════════════════════════
class ColumnIndex:
    """Index of the distinct values of ONE column."""

    def __init__(self, column: str, names: Iterable[Any], cfg: ResolverConfig,
                 aliases: Mapping | None = None) -> None:
        self.column = column
        self.cfg = cfg
        uniq = sorted({str(n).strip() for n in names if n is not None and str(n).strip()
                       and str(n).strip().lower() not in {"nan", "none", "null", "unknown"}})
        self.names: list[str] = uniq
        self.norm: list[str] = [normalize(n) for n in uniq]
        self.hint_words: set[str] = _label_words(column)
        self._exact: dict[str, list[int]] = {}
        self._token_ids: dict[str, set[int]] = {}
        self._acronym: dict[str, list[int]] = {}
        for i, nm in enumerate(self.norm):
            self._exact.setdefault(nm, []).append(i)
            toks = _tokens(nm)
            stripped = " ".join(t for t in toks if t not in _LEGAL_SUFFIX)
            if stripped and stripped != nm:
                self._exact.setdefault(stripped, []).append(i)
            for t in toks:
                self._token_ids.setdefault(t, set()).add(i)
            acr = "".join(t[0] for t in toks if t not in _LEGAL_SUFFIX and not t.isdigit())
            if len(acr) >= 2:
                self._acronym.setdefault(acr, []).append(i)
        self._sorted_tokens: list[str] = sorted(self._token_ids)

        # Generic words: universal + configured + learned from this column's data.
        self.generic: set[str] = set(UNIVERSAL_GENERIC_TOKENS) | {normalize(t) for t in cfg.generic_tokens}
        n = len(self.names)
        if n >= cfg.auto_generic_min_distinct:
            limit = max(cfg.auto_generic_min_count, int(n * cfg.auto_generic_ratio))
            self.generic |= {t for t, ids in self._token_ids.items() if len(ids) >= limit}
        # Question words that really occur inside this column's values are not stop words here.
        self.stop: frozenset[str] = frozenset(STOP_TOKENS - set(self._token_ids))
        self._alias: dict[str, list[int]] = self._build_aliases(aliases or {})

    def _build_aliases(self, aliases: Mapping) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for alias, targets in (aliases or {}).items():
            targets = [targets] if isinstance(targets, str) else list(targets or [])
            ids = sorted({i for t in targets for i in self._exact.get(normalize(t), [])})
            if ids:
                out[normalize(alias)] = ids
        return out

    # ── helpers ─────────────────────────────────────────────────────────────────
    def _prefix_ids(self, token: str) -> set[int]:
        lo = bisect.bisect_left(self._sorted_tokens, token)
        ids: set[int] = set()
        for t in self._sorted_tokens[lo:]:
            if not t.startswith(token):
                break
            ids |= self._token_ids[t]
        return ids

    def _make(self, kind: str, ids: list[int], raw: str) -> ColumnMatch | None:
        if not ids:
            return None
        values = [self.names[i] for i in ids]
        like = ""
        if kind in {"partial", "substring"}:
            key = normalize(raw)
            like = "%" + " ".join(t for t in _tokens(key) if t not in self.stop).upper() + "%"
        return ColumnMatch(self.column, kind, values[: self.cfg.max_values_per_mention], len(values), like)

    # ── resolution stages ───────────────────────────────────────────────────────
    def resolve_known(self, raw: str) -> ColumnMatch | None:
        """Exact or alias match only (very cheap; used by the n-gram scan)."""
        term = normalize(raw)
        if term in self._exact:
            return self._make("exact", list(self._exact[term]), raw)
        if term in self._alias:
            return self._make("alias", list(self._alias[term]), raw)
        return None

    def resolve_cheap(self, raw: str) -> ColumnMatch | None:
        """exact → alias → acronym → token-prefix → substring (no fuzzy)."""
        term = normalize(raw)
        if not term:
            return None
        known = self.resolve_known(raw)
        if known:
            return known

        toks = [t for t in _tokens(term) if t not in self.stop]
        significant = [t for t in toks if t not in self.generic]
        if not significant or all(t.isdigit() for t in significant):
            return None  # "hospital", "the clinic", "2024" … never match on their own
        if len(toks) == 1 and toks[0] in self._acronym and raw.strip().isupper():
            return self._make("acronym", list(self._acronym[toks[0]]), raw)

        # Token-prefix containment: every significant token is a prefix of some name token.
        if all(len(t) >= self.cfg.min_partial_len for t in significant):
            ids: set[int] | None = None
            for t in significant:
                hit = self._prefix_ids(t)
                ids = hit if ids is None else ids & hit
                if not ids:
                    break
            if ids:
                # Keep generic words the user typed ("Sun Hospital" ≠ "Sun Clinic").
                generic = [t for t in toks if t in self.generic]
                if generic:
                    narrowed = {i for i in ids if all(g in _tokens(self.norm[i]) for g in generic)}
                    ids = narrowed or ids
                return self._make("partial", sorted(ids), raw)

        # Substring inside a token ("rise" → "Sunrise …").
        joined = " ".join(significant)
        if len(joined) >= max(4, self.cfg.min_partial_len):
            sub = {i for i, nm in enumerate(self.norm) if joined in nm}
            if sub:
                return self._make("substring", sorted(sub), raw)
        return None

    def resolve_fuzzy(self, raw: str) -> ColumnMatch | None:
        """Typo tolerance. Slower, so callers run it only when nothing cheaper matched."""
        term = normalize(raw)
        toks = [t for t in _tokens(term) if t not in self.stop]
        significant = [t for t in toks if t not in self.generic]
        if not significant or all(t.isdigit() for t in significant):
            return None
        kind, ids = self._fuzzy(" ".join(toks))
        return self._make(kind, ids, raw) if ids else None

    def _fuzzy(self, term: str) -> tuple[str, list[int]]:
        if len(term) < 4 or not self.norm:
            return "", []
        cap = self.cfg.max_values_per_mention
        if process is not None:
            hits = process.extract(term, self.norm, scorer=fuzz.WRatio,
                                   score_cutoff=self.cfg.fuzzy_cutoff, limit=cap)
            # also token-level typo ("sunrize" vs token "sunrise")
            tok_hits = process.extract(term, self._sorted_tokens, scorer=fuzz.ratio,
                                       score_cutoff=self.cfg.fuzzy_cutoff, limit=5) if " " not in term else []
            ids = {h[2] for h in hits}
            for tok, _score, _ in tok_hits:
                if tok not in self.generic:
                    ids |= self._token_ids[tok]
            return ("fuzzy", sorted(ids)) if ids else ("", [])
        # difflib fallback on a candidate pool sharing a 2-char token prefix
        pool = sorted({i for t in term.split() if len(t) >= 2 for i in self._prefix_ids(t[:2])})
        cands = [self.norm[i] for i in pool] or self.norm[:5000]
        close = difflib.get_close_matches(term, cands, n=cap, cutoff=self.cfg.fuzzy_cutoff / 100)
        ids = sorted({i for c in close for i in self._exact.get(c, [])})
        return ("fuzzy", ids) if ids else ("", [])


# ═══════════════════════════════════════════════════════════════════════════════
# Whole dataset
# ═══════════════════════════════════════════════════════════════════════════════
class EntityResolver:
    """One :class:`ColumnIndex` per text column, plus question-level resolution."""

    def __init__(self, columns: Mapping[str, Iterable[Any]], cfg: ResolverConfig | None = None) -> None:
        self.cfg = cfg or ResolverConfig()
        self.indexes: dict[str, ColumnIndex] = {}
        for col, names in columns.items():
            idx = ColumnIndex(col, names, self.cfg, self._aliases_for(col))
            if idx.names:
                self.indexes[col] = idx
        words = sorted({w for idx in self.indexes.values() for w in idx.hint_words}, key=len, reverse=True)
        parts = list(_BASE_CUES)
        if words:
            parts.append(r"(?:" + "|".join(re.escape(w) for w in words) + r")(?:e?s)?")
        self._cue_re = re.compile(r"\b(?:" + "|".join(parts) + r")\b\s+", re.I)
        log.info("entity resolver ready: columns=%d [%s]", len(self.indexes),
                 ", ".join(f"{c}:{len(i.names)}" for c, i in self.indexes.items()))

    # ── construction helpers ────────────────────────────────────────────────────
    def _aliases_for(self, column: str) -> dict:
        merged: dict = {}
        for key, val in (self.cfg.aliases or {}).items():
            if isinstance(val, Mapping):
                if key == column:
                    merged.update(val)
            else:
                merged[key] = val
        return merged

    @staticmethod
    def collect_columns(df: pd.DataFrame, cfg: ResolverConfig) -> dict[str, list]:
        """Pick the columns to index and return ``{column: distinct values}``."""
        picked: dict[str, list] = {}
        explicit = [c for c in (cfg.columns or []) if c in df.columns]
        if cfg.columns:
            candidates = explicit
        else:
            pat = re.compile(cfg.exclude_name_pattern, re.I) if cfg.exclude_name_pattern else None
            skip = {str(c).casefold() for c in cfg.exclude_columns}
            candidates = []
            for col in df.columns:
                if str(col).casefold() in skip or (pat and pat.search(str(col))):
                    continue
                s = df[col]
                if not (is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)):
                    continue
                candidates.append(col)
        for col in candidates:
            try:
                vals = df[col].dropna().unique().tolist()
            except Exception:  # unhashable / odd column
                log.debug("entity resolver: skipping column %s", col)
                continue
            if not vals or len(vals) > cfg.max_distinct:
                continue
            if not cfg.columns:  # auto-detect filters (explicit columns bypass them)
                if len(vals) < cfg.min_distinct:
                    continue
                sample = vals[:2000]
                if sum(len(str(v)) for v in sample) / len(sample) > cfg.max_avg_len:
                    continue  # free text, not an entity
                if sum(1 for v in sample if _NUMERICISH_RE.fullmatch(str(v))) / len(sample) > 0.8:
                    continue  # numbers / dates stored as text
            picked[str(col)] = vals
        return picked

    @classmethod
    def from_frame(cls, df: pd.DataFrame, cfg: ResolverConfig | None = None) -> EntityResolver | None:
        cfg = cfg or ResolverConfig()
        if df is None or not len(getattr(df, "columns", [])):
            return None
        cols = cls.collect_columns(df, cfg)
        if not cols:
            return None
        resolver = cls(cols, cfg)
        return resolver if resolver.indexes else None

    # ── question-level extraction ───────────────────────────────────────────────
    def extract_mentions(self, question: str) -> list[str]:
        """Candidate entity mentions: quoted strings + cue-introduced name lists."""
        q = " ".join(str(question or "").split())
        found: list[str] = [m.group(1).strip() for m in _QUOTED_RE.finditer(q)]
        unquoted = _QUOTED_RE.sub(lambda m: f" {m.group(1)} ", q)
        consumed = 0
        for cue in self._cue_re.finditer(unquoted):
            if cue.start() < consumed:  # cue word inside a name already being read
                continue
            tail = unquoted[cue.end():]
            stop = _STOP_RE.search(tail)
            segment = tail[: stop.start()] if stop else tail
            consumed = cue.end() + len(segment)
            for part in _SPLIT_RE.split(segment):
                raw_words = part.strip(" -").split()
                words = list(raw_words)
                while words and words[0].lower() in STOP_TOKENS:
                    words.pop(0)  # "providers containing Sun" → "Sun"
                while words and words[-1].lower() in STOP_TOKENS:
                    words.pop()
                if words:
                    found.append(" ".join(words))
                if raw_words and len(raw_words) != len(words):
                    # untrimmed form, in case a "stop" word is really part of a value
                    found.append(" ".join(raw_words))
        out, seen = [], set()
        for m in found:
            k = normalize(m)
            if k and k not in seen:
                seen.add(k)
                out.append(m)
        return out[:12]

    # ── question-level resolution ───────────────────────────────────────────────
    def _rank(self, cands: list[ColumnMatch], q_words: set[str]) -> list[ColumnMatch]:
        """Keep only the best match level; prefer columns named in the question, then narrower matches."""
        if not cands:
            return []
        cands.sort(key=lambda c: (_TIER[c.match_type],
                                  not (self.indexes[c.column].hint_words & q_words),
                                  c.total_matches))
        best = _TIER[cands[0].match_type]
        return [c for c in cands if _TIER[c.match_type] == best][: self.cfg.max_columns_per_mention]

    def _resolve_mention(self, mention: str, q_words: set[str]) -> list[ColumnMatch]:
        cands = [m for idx in self.indexes.values() if (m := idx.resolve_cheap(mention))]
        if not cands:  # fuzzy only when no column matched anything cheaper
            cands = [m for idx in self.indexes.values() if (m := idx.resolve_fuzzy(mention))]
        return self._rank(cands, q_words)

    def _scan_exact(self, q_tokens: list[str], q_words: set[str], skip_keys: set[str]) -> list[MentionMatch]:
        """Find exact values / aliases written anywhere in the question (longest phrase first)."""
        n_tok = len(q_tokens)
        covered = [False] * n_tok
        out: list[MentionMatch] = []
        for size in range(min(self.cfg.ngram_max_tokens, n_tok), 0, -1):
            for i in range(n_tok - size + 1):
                if any(covered[i:i + size]):
                    continue
                gram = q_tokens[i:i + size]
                if all(t in STOP_TOKENS or t in UNIVERSAL_GENERIC_TOKENS for t in gram):
                    continue
                term = " ".join(gram)
                if len(term) < 3 or term in skip_keys:
                    continue
                cands = [m for idx in self.indexes.values() if (m := idx.resolve_known(term))]
                cands = self._rank(cands, q_words)
                if cands:
                    for j in range(i, i + size):
                        covered[j] = True
                    out.append(MentionMatch(term, cands))
        return out

    def resolve(self, question: str) -> list[MentionMatch]:
        q_tokens = _tokens(normalize(question))
        q_words = set(q_tokens) | {t[:-1] for t in q_tokens if t.endswith("s") and len(t) > 3}
        matches: list[MentionMatch] = []
        seen: set = set()
        keys: set[str] = set()

        def add(m: MentionMatch) -> None:
            sig = frozenset((c.column, tuple(c.values)) for c in m.candidates)
            if sig not in seen:
                seen.add(sig)
                matches.append(m)

        for mention in self.extract_mentions(question):
            key = normalize(mention)
            keys.add(key)
            cands = self._resolve_mention(mention, q_words)
            if cands:
                add(MentionMatch(mention, cands))
        if self.cfg.scan_exact_values:
            for m in self._scan_exact(q_tokens, q_words, keys):
                add(m)
        return matches[: self.cfg.max_mentions]

    # ── payload ─────────────────────────────────────────────────────────────────
    def augment(self, question: str, available_values: dict) -> dict:
        """Return ``available_values`` plus resolved values for this question.

        Returns the **same object, unchanged** when nothing resolves or when every
        mention is already one exact full value in one column.
        """
        try:
            matches = self.resolve(question)
        except Exception:  # resolution must never break answering
            log.exception("entity resolution failed; continuing without it")
            return available_values
        if not matches or all(
            len(m.candidates) == 1 and m.candidates[0].match_type == "exact"
            and len(m.candidates[0].values) == 1 for m in matches
        ):
            return available_values

        per_col: dict[str, list[str]] = {}
        mentions = []
        for m in matches:
            items = []
            for c in m.candidates:
                per_col.setdefault(c.column, []).extend(c.values)
                item: dict[str, Any] = {"column": c.column, "match_type": c.match_type,
                                        "resolved_values": c.values, "total_matches": c.total_matches}
                if c.truncated and c.like_pattern:
                    item["sql_condition"] = f"upper(trim(`{c.column}`)) LIKE '{_sql_literal(c.like_pattern)}'"
                items.append(item)
            mentions.append({"mention": m.text, "candidates": items})

        out = dict(available_values or {})
        for col, vals in per_col.items():
            vals = list(dict.fromkeys(vals))[: self.cfg.max_values_total]
            existing = out.get(col)
            if isinstance(existing, (list, tuple)):  # keep the original grounding values too
                have = set(vals)
                vals = vals + [v for v in existing if v not in have]
            out[col] = vals
        out["entity_resolution"] = {
            "columns": list(per_col),
            "mentions": mentions,
            "instructions": (
                "The user referred to dataset values by partial name, alias, abbreviation, typo or several "
                "names. They were resolved deterministically (no AI) to the exact database values listed in "
                "`mentions`. Each mention has `candidates`: the column and the exact values found in it. "
                "These values ARE valid filter values (do not mark the question INVALID_FILTER). "
                "If a mention has one candidate, filter on that column with `column` IN (...) using every "
                "resolved value verbatim. If it has several candidates, use the column that fits the meaning "
                "of the question (prefer a column the question names); do not combine different columns for "
                "one mention unless the question clearly asks for it. When a candidate has `sql_condition`, "
                "use that condition (OR-ed with the IN list) instead of listing values. When more than one "
                "value is resolved for a column, GROUP BY that column so each value is reported separately, "
                "unless the user asked for one combined figure. Ignore a mention that clearly is not about "
                "the values of any candidate column."
            ),
        }
        log.info("entity resolution: %d mention(s) → %d column(s) [%s]", len(matches), len(per_col),
                 ", ".join(f"{m.text}:{c.column}:{c.match_type}:{c.total_matches}"
                           for m in matches for c in m.candidates))
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Process-wide cache (one index set per loaded dataset)
# ═══════════════════════════════════════════════════════════════════════════════
class EntityResolverCache:
    """Builds the resolver lazily and rebuilds it only when the dataset changes."""

    def __init__(self, cfg: ResolverConfig | None = None) -> None:
        self.cfg = cfg or ResolverConfig()
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._resolver: EntityResolver | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> EntityResolverCache:
        cols = getattr(settings, "entity_resolution_columns", None)
        return cls(ResolverConfig(
            columns=list(cols) if cols else None,
            exclude_columns=list(getattr(settings, "entity_resolution_exclude_columns", []) or []),
            generic_tokens=list(getattr(settings, "entity_resolution_generic_tokens", []) or []),
            fuzzy_cutoff=float(getattr(settings, "entity_resolution_fuzzy_cutoff", 88.0) or 88.0),
            max_values_per_mention=int(getattr(settings, "entity_resolution_max_values", 25) or 25),
            aliases=dict(getattr(settings, "entity_aliases", {}) or {}),
        ))

    def get(self, df: pd.DataFrame | None) -> EntityResolver | None:
        if df is None or not len(getattr(df, "columns", [])):
            return None
        key = (id(df), len(df), tuple(str(c) for c in df.columns))
        with self._lock:
            if key != self._key:
                self._resolver = EntityResolver.from_frame(df, self.cfg)
                self._key = key
            return self._resolver

    def augment(self, question: str, df: pd.DataFrame | None, available_values: dict) -> dict:
        try:
            resolver = self.get(df)
        except Exception:
            log.exception("entity resolver could not be built; continuing without it")
            return available_values
        return resolver.augment(question, available_values) if resolver else available_values


__all__ = ["EntityResolver", "EntityResolverCache", "ResolverConfig", "MentionMatch", "ColumnMatch",
           "ColumnIndex", "normalize"]