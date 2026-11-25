# -*- coding: utf-8 -*-
"""Offline tests for scripts/daily_news_brief.py post-processing."""

import importlib.util
import json
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "daily_news_brief", Path(__file__).resolve().parents[1] / "scripts" / "daily_news_brief.py"
)
brief = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(brief)


class TestDailyNewsBrief(unittest.TestCase):
    def test_validate_items_drops_unknown_urls_and_respects_limit(self):
        raw = [
            {"headline": "A", "summary": "a", "url": "https://x/1"},
            {"headline": "B", "summary": "b", "url": "https://invented/2"},
            {"headline": "C", "summary": "c", "url": "https://x/3"},
            {"headline": "D", "summary": "d", "url": "https://x/1"},
        ]
        items = brief.validate_items(raw, {"https://x/1", "https://x/3"}, 2)
        self.assertEqual([i["headline"] for i in items], ["A", "C"])

    def test_validate_items_rejects_malformed_payload(self):
        self.assertEqual(brief.validate_items("nope", {"u"}, 3), [])
        self.assertEqual(brief.validate_items([{"headline": "", "summary": "s", "url": "u"}, 1], {"u"}, 3), [])

    def test_parse_json_block_strips_code_fence(self):
        data = brief.parse_json_block('```json\n{"kr": []}\n```')
        self.assertEqual(data, {"kr": []})
        with self.assertRaises(ValueError):
            brief.parse_json_block(json.dumps([1, 2]))

    def test_format_sections_skips_empty_regions(self):
        sources = {"kr": [{"title": "t", "url": "https://x/1", "content": ""}]}
        digest = {
            "kr": [{"headline": "제목", "summary": "요약", "url": "https://x/1"}],
            "us": [{"headline": "x", "summary": "y", "url": "https://x/9"}],
        }
        sections = brief.format_sections(digest, sources)
        self.assertEqual(len(sections), 1)
        self.assertIn("제목", sections[0])

    def test_chunk_message_splits_long_content(self):
        sections = ["s" * 2500, "t" * 2500]
        chunks = brief.chunk_message("H", sections, "F")
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[-1].endswith("F"))
        self.assertTrue(all(len(c) <= brief.TELEGRAM_LIMIT + 10 for c in chunks))


if __name__ == "__main__":
    unittest.main()
