"""Conversation History Compaction and ADK Context Caching utilities.

Prevents context window bloat during extended dealership negotiation sessions by:
1. Estimating token usage across turns (`estimate_tokens`)
2. Summarizing older negotiation turns into a compact, deterministic customer &
   deal-state digest (`summarize_older_turns`)
3. Applying sliding-window compaction (`compact_history`) while preserving the
   most recent turns intact
4. Providing Google ADK `EventsCompactionConfig` and `ContextCacheConfig` builders
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Union

from app.config import DEFAULT_CONFIG
from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR, PIIRedactor
from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import traced_operation


class ContextCompactor:
    """Sliding-window conversation compactor and structured turn summarizer."""

    _BUDGET_PATTERN = re.compile(
        r"\$\s?\d[\d,]*(?:\.\d{2})?(?:\s*/\s*mo(?:nth)?|\s*per\s*month|\s*down|\s*budget)?",
        re.IGNORECASE,
    )
    _CREDIT_SCORE_PATTERN = re.compile(
        r"\b([3-8]\d{2})\s*(?:credit|fico|score)\b|\b(?:credit\s*score|fico)\s*(?:of|is|:)?\s*([3-8]\d{2})\b",
        re.IGNORECASE,
    )
    _SEATING_PATTERN = re.compile(
        r"\b([2-3]-row|[5-8]\s*(?:seat|passenger)s?|family\s*of\s*[3-8])\b",
        re.IGNORECASE,
    )
    _VEHICLE_KEYWORDS = (
        "suv",
        "crossover",
        "sedan",
        "truck",
        "ev",
        "hybrid",
        "phev",
        "highlander",
        "grand highlander",
        "rav4",
        "cr-v",
        "pilot",
        "telluride",
        "sorento",
        "santa fe",
        "cx-90",
        "cx-50",
        "camry",
        "accord",
        "corolla",
        "civic",
        "elantra",
        "ioniq",
        "ev9",
        "equinox",
        "f-150",
        "tacoma",
        "maverick",
        "outback",
        "trade-in",
        "trade in",
    )

    def __init__(
        self,
        max_turns: int = DEFAULT_CONFIG.max_history_turns,
        max_token_budget: int = DEFAULT_CONFIG.max_token_budget,
        keep_recent_turns: int = 6,
        redactor: Optional[PIIRedactor] = None,
    ) -> None:
        self.max_turns = max_turns
        self.max_token_budget = max_token_budget
        self.keep_recent_turns = keep_recent_turns
        self.redactor = redactor or DEFAULT_PII_REDACTOR

    @staticmethod
    def estimate_tokens(
        turns_or_text: Union[str, Sequence[Dict[str, Any]]],
    ) -> int:
        """Estimates token count using a deterministic ~4 chars/token heuristic.

        Args:
            turns_or_text: Raw string or sequence of conversation turn dictionaries.

        Returns:
            Estimated integer token count.
        """
        if isinstance(turns_or_text, str):
            return max(1, len(turns_or_text) // 4) if turns_or_text else 0

        total_chars = 0
        for turn in turns_or_text:
            role = str(turn.get("role", ""))
            content = str(turn.get("content", ""))
            total_chars += len(role) + len(content) + 8
        return total_chars // 4

    @traced_operation("autodeal.memory.summarize_older_turns")
    def summarize_older_turns(self, turns: Sequence[Dict[str, Any]]) -> str:
        """Compiles older conversation turns into a dense, PII-free deal summary.

        Preserves critical sales context (budget figures, credit score, seating/family
        needs, mentioned models/fuel types, and trade-in details) while discarding
        repetitive conversational filler.

        Args:
            turns: Sequence of older turn dictionaries to summarize.

        Returns:
            Structured summary string suitable for injection as a system context header.
        """
        if not turns:
            return "[COMPACTED_HISTORY_SUMMARY] No prior turns."

        combined_lines: List[str] = []
        budgets: List[str] = []
        credit_scores: List[str] = []
        seating_notes: List[str] = []
        matched_keywords: List[str] = []

        for turn in turns:
            clean_text = self.redactor.redact_text(str(turn.get("content", "")))
            if not clean_text.strip():
                continue
            combined_lines.append(
                f"{turn.get('role', 'user')}: {clean_text.strip()[:160]}"
            )

            for m in self._BUDGET_PATTERN.findall(clean_text):
                val = m.strip()
                if val and val not in budgets:
                    budgets.append(val)

            for m_tuple in self._CREDIT_SCORE_PATTERN.findall(clean_text):
                score = m_tuple[0] or m_tuple[1]
                if score and score not in credit_scores:
                    credit_scores.append(score)

            for s_match in self._SEATING_PATTERN.findall(clean_text):
                if s_match.lower() not in seating_notes:
                    seating_notes.append(s_match.lower())

            lower_text = clean_text.lower()
            for kw in self._VEHICLE_KEYWORDS:
                if kw in lower_text and kw not in matched_keywords:
                    matched_keywords.append(kw)

        highlights: List[str] = [f"turns_compacted={len(turns)}"]
        if budgets:
            highlights.append(f"budget_mentions={', '.join(budgets[:5])}")
        if credit_scores:
            highlights.append(f"credit_score={', '.join(credit_scores[:2])}")
        if seating_notes:
            highlights.append(f"seating_needs={', '.join(seating_notes[:3])}")
        if matched_keywords:
            highlights.append(f"vehicle_interests={', '.join(matched_keywords[:10])}")

        digest_header = "[COMPACTED_HISTORY_SUMMARY] " + " | ".join(highlights)
        key_excerpts = " ; ".join(combined_lines[:4])
        full_summary = f"{digest_header} || Key Excerpts: {key_excerpts}"
        return self.redactor.redact_text(full_summary)

    @traced_operation("autodeal.memory.compact_history")
    def compact_history(
        self,
        turns: Sequence[Dict[str, Any]],
        max_turns: Optional[int] = None,
        max_tokens: Optional[int] = None,
        keep_recent: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compacts conversation history when turn count or token budget is exceeded.

        Args:
            turns: Full chronological list of session turn dictionaries.
            max_turns: Override for maximum allowed turns before compaction triggers.
            max_tokens: Override for maximum allowed tokens before compaction triggers.
            keep_recent: Number of most-recent turns to keep verbatim.

        Returns:
            Compacted list of turn dictionaries where older turns are replaced by
            a single structured summary turn (`role='system'`).
        """
        turn_limit = max_turns if max_turns is not None else self.max_turns
        token_limit = max_tokens if max_tokens is not None else self.max_token_budget
        recent_count = (
            keep_recent if keep_recent is not None else self.keep_recent_turns
        )

        turns_list = [dict(t) for t in turns]
        initial_tokens = self.estimate_tokens(turns_list)

        log_intent(
            action_name="compact_history",
            intended_arguments={
                "input_turns": len(turns_list),
                "initial_tokens": initial_tokens,
                "turn_limit": turn_limit,
                "token_limit": token_limit,
            },
            agent_name="dealership_coordinator",
        )

        needs_compaction = (
            len(turns_list) > turn_limit or initial_tokens > token_limit
        ) and len(turns_list) > recent_count

        if not needs_compaction:
            sanitized_turns = [
                {
                    **t,
                    "content": self.redactor.redact_text(str(t.get("content", ""))),
                }
                for t in turns_list
            ]
            log_outcome(
                action_name="compact_history",
                status="no_op",
                actual_outcome={
                    "compacted": False,
                    "output_turns": len(sanitized_turns),
                    "output_tokens": initial_tokens,
                },
                agent_name="dealership_coordinator",
            )
            return sanitized_turns

        older_slice = turns_list[:-recent_count] if recent_count > 0 else turns_list
        recent_slice = turns_list[-recent_count:] if recent_count > 0 else []

        summary_text = self.summarize_older_turns(older_slice)
        summary_turn: Dict[str, Any] = {
            "role": "system",
            "content": summary_text,
            "metadata": {
                "is_compacted_summary": True,
                "compacted_turn_count": len(older_slice),
            },
        }

        sanitized_recent = [
            {
                **t,
                "content": self.redactor.redact_text(str(t.get("content", ""))),
            }
            for t in recent_slice
        ]

        compacted = [summary_turn, *sanitized_recent]
        final_tokens = self.estimate_tokens(compacted)

        log_outcome(
            action_name="compact_history",
            status="success",
            actual_outcome={
                "compacted": True,
                "input_turns": len(turns_list),
                "output_turns": len(compacted),
                "initial_tokens": initial_tokens,
                "output_tokens": final_tokens,
            },
            agent_name="dealership_coordinator",
        )
        return compacted


def build_adk_events_compaction_config(
    compaction_interval: int = 6,
    overlap_size: int = 2,
) -> Any:
    """Builds a Google ADK `EventsCompactionConfig` for the root `App`.

    Args:
        compaction_interval: Number of new invocations before event compaction runs.
        overlap_size: Number of overlapping recent events retained for continuity.

    Returns:
        Configured `EventsCompactionConfig` instance (or fallback config dictionary).
    """
    try:
        from google.adk.apps.app import EventsCompactionConfig

        return EventsCompactionConfig(
            compaction_interval=compaction_interval,
            overlap_size=overlap_size,
        )
    except Exception:
        return {
            "compaction_interval": compaction_interval,
            "overlap_size": overlap_size,
        }


def build_adk_context_cache_config(
    min_tokens: int = 1024,
    ttl_seconds: int = 1800,
    cache_intervals: int = 5,
) -> Any:
    """Builds a Google ADK `ContextCacheConfig` for caching static prompts & schemas.

    Args:
        min_tokens: Minimum token threshold before context caching activates.
        ttl_seconds: Time-to-live in seconds for cached context prefixes.
        cache_intervals: Maximum invocations before refreshing the cache.

    Returns:
        Configured `ContextCacheConfig` instance (or fallback config dictionary).
    """
    try:
        from google.adk.agents.context_cache_config import ContextCacheConfig

        return ContextCacheConfig(
            min_tokens=min_tokens,
            ttl_seconds=ttl_seconds,
            cache_intervals=cache_intervals,
        )
    except Exception:
        return {
            "min_tokens": min_tokens,
            "ttl_seconds": ttl_seconds,
            "cache_intervals": cache_intervals,
        }


DEFAULT_CONTEXT_COMPACTOR = ContextCompactor()
