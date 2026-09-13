"""Covers the Step-1.5 profiling response cache.

Profiling trains the whole selector on a 20-50 pair sample, and re-querying identical
prompts at temperature=0 was measured to change 8 of 20 pairs. Without a cache, two runs
that differ only by --selection-policy also differ in what the selector learned, which
makes the comparison unreadable. These tests pin the cache's identity rules: same prompt
replays, and anything that should produce a genuinely different answer must miss.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from code.attribute_selection import llm_guided


class ProfileCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache_dir = Path(tempfile.mkdtemp())
        llm_guided.set_profile_cache_dir(self.cache_dir)
        llm_guided.set_profile_cache_enabled(True)
        self.calls = []

    def tearDown(self):
        llm_guided.set_profile_cache_enabled(True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def fake_completion(self, content='{"attribute_importance": {"name": 3}}'):
        def _call(messages, temperature=0, max_tokens=None):
            self.calls.append(messages[0]["content"])
            return content, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return _call

    def run_profiling(self, prompt, scoring="ordinal", model="model-a", content=None):
        kwargs = {"content": content} if content else {}
        with mock.patch.object(llm_guided, "create_chat_completion_text", self.fake_completion(**kwargs)), \
             mock.patch.object(llm_guided, "get_llm_model", return_value=model):
            return llm_guided.profiling_completion(prompt, scoring)

    def test_first_call_queries_and_second_replays(self):
        first_content, first_usage = self.run_profiling("PROMPT A")
        second_content, second_usage = self.run_profiling("PROMPT A")

        self.assertEqual(len(self.calls), 1, "second call must not reach the API")
        self.assertEqual(first_content, second_content)
        self.assertFalse(first_usage["cache_hit"])
        self.assertTrue(second_usage["cache_hit"])

    def test_replayed_usage_reports_the_original_token_counts(self):
        _, first = self.run_profiling("PROMPT A")
        _, second = self.run_profiling("PROMPT A")
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertEqual(first[field], second[field])

    def test_a_different_prompt_misses(self):
        # A different seed samples different profile pairs, so their prompts differ and
        # must still be queried -- the cache must not flatten seed variation.
        self.run_profiling("PROMPT A")
        _, usage = self.run_profiling("PROMPT B")
        self.assertFalse(usage["cache_hit"])
        self.assertEqual(len(self.calls), 2)

    def test_a_different_scoring_mode_misses(self):
        self.run_profiling("PROMPT A", scoring="ordinal")
        _, usage = self.run_profiling("PROMPT A", scoring="decisive")
        self.assertFalse(usage["cache_hit"], "scoring mode is the variable under test")

    def test_a_different_model_misses(self):
        self.run_profiling("PROMPT A", model="model-a")
        _, usage = self.run_profiling("PROMPT A", model="model-b")
        self.assertFalse(usage["cache_hit"], "responses are not transferable across models")

    def test_disabling_the_cache_always_queries(self):
        self.run_profiling("PROMPT A")
        llm_guided.set_profile_cache_enabled(False)
        _, usage = self.run_profiling("PROMPT A")
        self.assertFalse(usage["cache_hit"])
        self.assertEqual(len(self.calls), 2)

    def test_disabled_cache_does_not_write(self):
        llm_guided.set_profile_cache_enabled(False)
        self.run_profiling("PROMPT A")
        self.assertEqual(list(self.cache_dir.glob("*.json")), [])

    def test_corrupt_cache_entry_falls_back_to_querying(self):
        self.run_profiling("PROMPT A")
        for entry in self.cache_dir.glob("*.json"):
            entry.write_text("{not json")
        _, usage = self.run_profiling("PROMPT A")
        self.assertFalse(usage["cache_hit"])
        self.assertEqual(len(self.calls), 2)

    def test_cache_entry_records_what_produced_it(self):
        self.run_profiling("PROMPT A", scoring="decisive", model="model-x")
        entry = json.loads(next(self.cache_dir.glob("*.json")).read_text())
        self.assertEqual(entry["scoring"], "decisive")
        self.assertEqual(entry["model"], "model-x")
        self.assertEqual(entry["prompt"], "PROMPT A")

    def test_key_is_stable_across_calls(self):
        key_one = llm_guided.profile_cache_key("PROMPT A", "model-a", "ordinal")
        key_two = llm_guided.profile_cache_key("PROMPT A", "model-a", "ordinal")
        self.assertEqual(key_one, key_two)
        self.assertNotEqual(key_one, llm_guided.profile_cache_key("PROMPT A", "model-a", "decisive"))


if __name__ == "__main__":
    unittest.main()
