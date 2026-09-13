import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from code import matcher
from code import main
from code.llm_client import CompletionError
from code.main import check_api_errors
from code.matcher_checkpoint import MatcherCheckpoint


def completion(answer="Yes"):
    return answer, {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11,
                    "provider": "test-provider", "served_model": "test-model"}


class MatcherCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "draw1"
        self.df = pd.DataFrame([{"title": "same", "year": "2000"}] * 2)
        self.pairs = [(0, 0), (1, 1)]
        self.addCleanup(matcher.set_cache_enabled, matcher.USE_CACHE)
        matcher.set_cache_enabled(False)
        self.model_patch = patch.object(matcher, "get_llm_model", return_value="test-model")
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.routing_patch = patch.object(matcher, "request_routing_options", return_value={})
        self.routing_patch.start()
        self.addCleanup(self.routing_patch.stop)

    def run_matcher(self, **kwargs):
        return matcher.infer_candidates_pairwise(
            kwargs.pop("df_a", self.df), self.df, self.pairs, max_workers=1,
            checkpoint_dir=kwargs.pop("directory", self.directory), **kwargs,
        )

    def test_identical_content_on_distinct_pairs_still_makes_independent_calls(self):
        with patch.object(matcher, "create_chat_completion_text", side_effect=[completion("Yes"), completion("No")]) as api:
            first = self.run_matcher()
        self.assertEqual(api.call_count, 2)
        self.assertEqual(first.answer.tolist(), ["Yes", "No"])
        with patch.object(matcher, "create_chat_completion_text") as api:
            resumed = self.run_matcher()
        api.assert_not_called()
        self.assertTrue(resumed.checkpoint_hit.all())
        self.assertFalse(resumed.cache_hit.any())
        self.assertEqual(resumed.total_tokens.sum(), first.total_tokens.sum())
        self.assertEqual(resumed.provider.tolist(), first.provider.tolist())
        self.assertEqual(resumed.raw_response.tolist(), ["Yes", "No"])

    def test_new_draw_directory_does_not_replay_another_draw(self):
        with patch.object(matcher, "create_chat_completion_text", return_value=completion()) as api:
            self.run_matcher()
            self.run_matcher(directory=Path(self.tmp.name) / "draw2")
        self.assertEqual(api.call_count, 4)

    def test_interrupt_preserves_completed_pairs_and_resume_only_calls_missing_pair(self):
        with patch.object(matcher, "create_chat_completion_text", side_effect=[completion(), KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                self.run_matcher()
        self.assertEqual(len(list(self.directory.glob("pair_*.json"))), 1)
        with patch.object(matcher, "create_chat_completion_text", return_value=completion("No")) as api:
            resumed = self.run_matcher()
        api.assert_called_once()
        self.assertEqual(resumed.answer.tolist(), ["Yes", "No"])
        self.assertEqual(resumed.checkpoint_hit.tolist(), [True, False])

    def test_failed_pairs_are_saved_but_not_replayed_and_usage_is_retained(self):
        error = CompletionError("empty response", {
            "prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9,
            "attempts": [{"error": "empty", "total_tokens": 9}],
        })
        with patch.object(matcher, "create_chat_completion_text", side_effect=[completion(), error]):
            failed = self.run_matcher()
        with self.assertRaises(RuntimeError):
            check_api_errors(failed)
        self.assertEqual(len(list(self.directory.glob("pair_*.json"))), 2)
        with patch.object(matcher, "create_chat_completion_text", return_value=completion("No")) as api:
            resumed = self.run_matcher()
        api.assert_called_once()
        self.assertEqual(check_api_errors(resumed), 0)
        self.assertEqual(resumed.total_tokens.sum(), 31)
        self.assertEqual(resumed.iloc[1].attempts, [{"error": "empty", "total_tokens": 9}])

    def test_changed_data_cannot_reuse_checkpoint(self):
        with patch.object(matcher, "create_chat_completion_text", return_value=completion()):
            self.run_matcher()
        changed = self.df.copy()
        changed.loc[0, "title"] = "different"
        with patch.object(matcher, "create_chat_completion_text") as api:
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.run_matcher(df_a=changed)
        api.assert_not_called()

    def test_changed_field_order_cannot_reuse_checkpoint(self):
        with patch.object(matcher, "create_chat_completion_text", return_value=completion()):
            self.run_matcher(selected_attributes_by_pair={p: ["title", "year"] for p in self.pairs})
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.run_matcher(selected_attributes_by_pair={p: ["year", "title"] for p in self.pairs})

    def test_changed_routing_cannot_reuse_checkpoint(self):
        with patch.object(matcher, "create_chat_completion_text", return_value=completion()):
            self.run_matcher()
        with patch.object(matcher, "request_routing_options", return_value={"provider": {"order": ["other"]}}):
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.run_matcher()

    def test_response_cache_must_be_disabled_for_independent_draws(self):
        matcher.set_cache_enabled(True)
        with self.assertRaisesRegex(ValueError, "disable-matcher-cache"):
            self.run_matcher()

    def test_corrupt_completed_result_fails_loudly_instead_of_rebilling(self):
        with patch.object(matcher, "create_chat_completion_text", return_value=completion()):
            self.run_matcher()
        path = next(self.directory.glob("pair_*.json"))
        with path.open("w") as stream:
            stream.write("{")
        with patch.object(matcher, "create_chat_completion_text") as api:
            with self.assertRaises(json.JSONDecodeError):
                self.run_matcher()
        api.assert_not_called()

    def test_concurrent_draw_writer_is_rejected(self):
        with MatcherCheckpoint(self.directory, {"test": 1}):
            with self.assertRaises(BlockingIOError):
                with MatcherCheckpoint(self.directory, {"test": 1}):
                    self.fail("second writer acquired the checkpoint")

    def test_pipeline_resume_preserves_metrics_tokens_and_provider_accounting(self):
        import numpy as np

        config = {**main.DATASET_CONFIGS["DBLP-Scholar"],
                  "cache_dir": str(Path(self.tmp.name) / "cache")}
        usage = completion()[1]
        usage["attempts"] = [{"prompt_tokens": 10, "completion_tokens": 1,
                              "total_tokens": 11, "provider": "test-provider",
                              "started_at": "2026-09-07T03:00:00+00:00",
                              "finished_at": "2026-09-07T03:00:01+00:00"}]
        with patch.dict(main.DATASET_CONFIGS, {"DBLP-Scholar": config}), \
             patch.object(main, "LOG_ROOT", Path(self.tmp.name) / "logs"), \
             patch.object(main, "load_configured_dataset", return_value=(self.df, self.df, {(0, 0)})), \
             patch.object(main, "SentenceTransformer"), \
             patch.object(main, "run_lsh_blocking", return_value=(np.eye(2), np.eye(2), self.pairs, np.ones(2))), \
             patch.object(matcher, "create_chat_completion_text", return_value=("Yes", usage)) as api:
            for _ in range(2):
                main.run_pipeline("DBLP-Scholar", experiment_mode="full", use_matcher_cache=False,
                                  matcher_checkpoint_dir=self.directory)
        self.assertEqual(api.call_count, 2)
        paths = sorted((Path(self.tmp.name) / "logs").glob("*/runs/*/run_summary.json"))
        first, second = [json.loads(p.read_text()) for p in paths]
        for key in ("candidate_f1", "end_to_end_f1", "total_tokens", "providers", "request_attempts"):
            self.assertEqual(first["matching"][key], second["matching"][key])
        self.assertEqual(second["matching"]["checkpoint_hits"], 2)
        self.assertEqual(second["matching"]["cache_hits"], 0)
        self.assertEqual(second["matching"]["providers"], {"test-provider": 2})
        csv = pd.read_csv(paths[-1].parent / "final_results.csv")
        self.assertEqual(json.loads(csv.iloc[0].attempts)[0]["provider"], "test-provider")


if __name__ == "__main__":
    unittest.main()
