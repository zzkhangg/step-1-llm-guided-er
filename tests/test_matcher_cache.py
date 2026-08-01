import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from code import matcher


class MatcherCacheTests(unittest.TestCase):
    def setUp(self):
        self.df_a = pd.DataFrame([{"name": "A"}])
        self.df_b = pd.DataFrame([{"name": "A"}])

    def test_matcher_cache_can_be_disabled(self):
        calls = []

        def fake_completion(*args, **kwargs):
            calls.append((args, kwargs))
            return "Yes", {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}

        with tempfile.TemporaryDirectory() as tmpdir:
            matcher.set_cache_dir(tmpdir)
            matcher.set_cache_enabled(False)
            with patch("code.matcher.create_chat_completion_text", side_effect=fake_completion):
                first = matcher.infer_pair(0, 0, self.df_a, self.df_b)
                second = matcher.infer_pair(0, 0, self.df_a, self.df_b)

            self.assertFalse(first["cache_hit"])
            self.assertFalse(second["cache_hit"])
            self.assertEqual(len(calls), 2)
            self.assertEqual(list(Path(tmpdir).glob("*.json")), [])

    def test_matcher_cache_enabled_reuses_saved_result(self):
        calls = []

        def fake_completion(*args, **kwargs):
            calls.append((args, kwargs))
            return "Yes", {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}

        with tempfile.TemporaryDirectory() as tmpdir:
            matcher.set_cache_dir(tmpdir)
            matcher.set_cache_enabled(True)
            with patch("code.matcher.create_chat_completion_text", side_effect=fake_completion):
                first = matcher.infer_pair(0, 0, self.df_a, self.df_b)
                second = matcher.infer_pair(0, 0, self.df_a, self.df_b)

            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(list(Path(tmpdir).glob("*.json"))), 1)
            self.assertEqual(first["prompt_hash"], matcher.prompt_hash())
            self.assertEqual(second["prompt_hash"], matcher.prompt_hash())

    def test_matcher_cache_key_includes_prompt_hash(self):
        rec_a = {"name": "A"}
        rec_b = {"name": "A"}

        first_hash = matcher.record_pair_hash(
            rec_a,
            rec_b,
            model="test-model",
            prompt_template="Prompt version one",
        )
        second_hash = matcher.record_pair_hash(
            rec_a,
            rec_b,
            model="test-model",
            prompt_template="Prompt version two",
        )

        self.assertNotEqual(first_hash, second_hash)


if __name__ == "__main__":
    unittest.main()
