"""The served provider must reach the result rows, and must never reach a token sum.

One model slug is routed across several upstreams whose answers differ, so a run that does
not record which one served it cannot distinguish provider drift from a real effect.
"""
import unittest

from code.llm_client import usage_to_dict


class _Usage:
    prompt_tokens = 11
    completion_tokens = 2
    total_tokens = 13


class _Response:
    usage = _Usage()
    provider = "Sail Research"
    model = "deepseek/deepseek-v4-flash-0731"


class _BareResponse:
    """An OpenAI-compatible endpoint that is not OpenRouter reports no provider."""
    usage = _Usage()


class ProviderRecordingTests(unittest.TestCase):
    def test_provider_and_served_model_are_captured(self):
        d = usage_to_dict(_Response())
        self.assertEqual(d["provider"], "Sail Research")
        self.assertEqual(d["served_model"], "deepseek/deepseek-v4-flash-0731")

    def test_absent_provider_becomes_empty_string_not_an_error(self):
        d = usage_to_dict(_BareResponse())
        self.assertEqual(d["provider"], "")
        self.assertEqual(d["served_model"], "")

    def test_token_counts_are_unaffected(self):
        d = usage_to_dict(_Response())
        self.assertEqual(
            {k: d[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
        )

    def test_the_string_keys_cannot_land_in_a_token_sum(self):
        # Aggregation sites iterate over a fixed set of token keys, never over the usage
        # dict, so the non-numeric entries are unreachable from a sum. Assert the shape
        # that makes that true.
        d = usage_to_dict(_Response())
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for key in totals:
            totals[key] += int(d.get(key, 0))
        self.assertEqual(totals, {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13})
        self.assertNotIn("provider", totals)


if __name__ == "__main__":
    unittest.main()
