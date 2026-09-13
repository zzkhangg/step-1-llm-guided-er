"""Guards against recording metrics from a run whose LLM calls failed.

A failed call is recorded as ``answer="Error"`` and scored as a non-match, so a run
that lost its API budget still produces a complete run summary: F1 of 0.0000 if every
call failed, some plausible-looking number if only some did. Nine such runs were
mistaken for results before the pipeline refused to write them.
"""

import inspect
import unittest
from unittest.mock import patch

import pandas as pd

import code.main as main


def result_frame(n_pairs, n_errors):
    """A matcher result frame with the first n_errors rows marked as failed calls."""
    rows = []
    for position in range(n_pairs):
        failed = position < n_errors
        rows.append({
            "indexA": position,
            "indexB": position,
            "answer": "Error" if failed else "No",
            "llm_error": "402 Insufficient credits" if failed else "",
        })
    return pd.DataFrame(rows)


class CheckApiErrorsTests(unittest.TestCase):
    def test_clean_run_passes_and_reports_zero(self):
        self.assertEqual(main.check_api_errors(result_frame(4, n_errors=0)), 0)

    def test_total_failure_raises_by_default(self):
        with self.assertRaises(RuntimeError) as ctx:
            main.check_api_errors(result_frame(4, n_errors=4))
        self.assertIn("4/4", str(ctx.exception))
        self.assertIn("100.0%", str(ctx.exception))
        self.assertIn("Nothing has been written", str(ctx.exception))

    def test_partial_failure_also_raises(self):
        # The 28.9%-failed run reported a plausible 0.7672 F1; that must not happen.
        with self.assertRaises(RuntimeError) as ctx:
            main.check_api_errors(result_frame(100, n_errors=29))
        self.assertIn("29/100", str(ctx.exception))

    def test_a_single_failure_is_enough_to_raise(self):
        with self.assertRaises(RuntimeError):
            main.check_api_errors(result_frame(1000, n_errors=1))

    def test_override_returns_the_count_instead_of_raising(self):
        count = main.check_api_errors(result_frame(4, n_errors=3), allow_api_errors=True)
        self.assertEqual(count, 3)

    def test_missing_answer_column_is_tolerated(self):
        self.assertEqual(main.check_api_errors(pd.DataFrame([{"indexA": 0}])), 0)

    def test_error_count_agrees_with_the_llm_error_column(self):
        # The sweep driver reads api_errors; the two signals must not diverge.
        frame = result_frame(10, n_errors=4)
        self.assertEqual(int((frame["answer"] == "Error").sum()),
                         int((frame["llm_error"] != "").sum()))


class GuardWiringTests(unittest.TestCase):
    """The guard must stay wired into the pipeline entry point and the CLI."""

    def test_run_pipeline_guards_by_default(self):
        params = inspect.signature(main.run_pipeline).parameters
        self.assertIn("allow_api_errors", params)
        self.assertIs(params["allow_api_errors"].default, False)

    def test_cli_defaults_to_guarding(self):
        with patch("sys.argv", ["main.py"]):
            self.assertFalse(main.parse_args().allow_api_errors)

    def test_cli_exposes_the_override(self):
        with patch("sys.argv", ["main.py", "--allow-api-errors"]):
            self.assertTrue(main.parse_args().allow_api_errors)


if __name__ == "__main__":
    unittest.main()
