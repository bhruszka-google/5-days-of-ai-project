"""Unit tests for app/memory/compaction.py."""

from __future__ import annotations

from app.memory.compaction import (
    ContextCompactor,
    build_adk_context_cache_config,
    build_adk_events_compaction_config,
)


class TestContextCompactor:
    """Tests for token estimation, sliding-window compaction, and ADK config builders."""

    def test_estimate_tokens_for_strings_and_turn_lists(self) -> None:
        compactor = ContextCompactor()
        assert compactor.estimate_tokens("") == 0
        assert compactor.estimate_tokens("Four score and seven years ago") > 0

        turns = [
            {"role": "user", "content": "Looking for a 3-row hybrid SUV under $650/mo"},
            {
                "role": "assistant",
                "content": "Checking our Highlander Hybrid inventory.",
            },
        ]
        assert compactor.estimate_tokens(turns) > 15

    def test_sliding_window_compaction_preserves_recent_turns_and_key_facts(
        self,
    ) -> None:
        compactor = ContextCompactor(
            max_turns=5, max_token_budget=500, keep_recent_turns=2
        )
        turns = [
            {
                "role": "user",
                "content": "Customer has a family of 7, needs a 3-row hybrid SUV under $650/mo with $4,000 down.",
            },
            {
                "role": "assistant",
                "content": "We have the 2026 Toyota Highlander Hybrid XLE and Kia Sorento Hybrid in stock.",
            },
            {
                "role": "user",
                "content": "Customer credit score is 745 and trading in a 2021 Honda CR-V. Email: secret@buyer.com.",
            },
            {
                "role": "assistant",
                "content": "The 2021 Honda CR-V appraises at $21,500 in good condition.",
            },
            {
                "role": "user",
                "content": "How does the Highlander compare to the Grand Highlander?",
            },
            {
                "role": "assistant",
                "content": "The Grand Highlander offers extra 3rd-row legroom and 8-passenger seating.",
            },
        ]

        compacted = compactor.compact_history(turns)
        # 1 summary turn + 2 kept recent turns = 3 turns total
        assert len(compacted) == 3
        summary_turn = compacted[0]
        assert summary_turn["role"] == "system"
        assert summary_turn["metadata"]["is_compacted_summary"] is True
        assert summary_turn["metadata"]["compacted_turn_count"] == 4
        assert "[COMPACTED_HISTORY_SUMMARY]" in summary_turn["content"]
        assert "$650/mo" in summary_turn["content"]
        assert "745" in summary_turn["content"]
        assert "highlander" in summary_turn["content"].lower()
        assert "secret@buyer.com" not in summary_turn["content"]

        # Most recent 2 turns preserved verbatim
        assert compacted[1]["content"] == turns[-2]["content"]
        assert compacted[2]["content"] == turns[-1]["content"]

    def test_no_op_when_below_thresholds(self) -> None:
        compactor = ContextCompactor(
            max_turns=10, max_token_budget=5000, keep_recent_turns=4
        )
        short_turns = [
            {"role": "user", "content": "Show me compact sedans under $30,000."},
            {"role": "assistant", "content": "Here is the 2025 Toyota Corolla LE."},
        ]
        result = compactor.compact_history(short_turns)
        assert len(result) == 2
        assert result[0]["role"] == "user"

    def test_adk_compaction_and_cache_config_builders(self) -> None:
        compaction_cfg = build_adk_events_compaction_config(
            compaction_interval=6, overlap_size=2
        )
        assert compaction_cfg is not None
        cache_cfg = build_adk_context_cache_config(min_tokens=1024, ttl_seconds=1800)
        assert cache_cfg is not None
