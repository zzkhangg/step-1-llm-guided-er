"""Covers the escalating retry for responses cut off by max_tokens.

A minority of OpenRouter's providers ignore reasoning.enabled=false and spend the token
budget on hidden thinking, returning no content. Re-sending the identical cap just
re-truncates, so one such call exhausted its retries and voided a 25-minute DBLP-ACM run
(1 failure in 13,080 calls is enough, because the pipeline refuses to record a run with
any failed call). These tests pin the escalation that covers that provider.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from code import llm_client


def response(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, refusal=None),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


TRUNCATED = lambda: response(None, finish_reason="length")
MSGS = [{"role": "user", "content": "hi"}]


class TruncationRetryTests(unittest.TestCase):
    def call(self, responses, **kwargs):
        """Run create_chat_completion_text against a scripted sequence of responses."""
        self.caps = []
        it = iter(responses)

        def create(model, messages, temperature, max_tokens, extra_body):
            self.caps.append(max_tokens)
            return next(it)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with mock.patch.object(llm_client, "get_llm_client", return_value=client), \
             mock.patch.object(llm_client, "get_llm_model", return_value="m"), \
             mock.patch.object(llm_client.time, "sleep", lambda *_: None):
            return llm_client.create_chat_completion_text(MSGS, **kwargs)

    def test_truncated_then_success_raises_the_cap(self):
        text, _ = self.call([TRUNCATED(), response("Yes")], max_tokens=256)
        self.assertEqual(text, "Yes")
        self.assertEqual(self.caps, [256, 256 * llm_client.TRUNCATION_RETRY_FACTOR])

    def test_cap_escalates_on_every_truncation(self):
        text, _ = self.call([TRUNCATED(), TRUNCATED(), response("No")], max_tokens=256)
        self.assertEqual(text, "No")
        self.assertEqual(self.caps, [256, 1024, 4096])

    def test_escalation_stops_at_the_ceiling(self):
        with self.assertRaises(RuntimeError):
            self.call([TRUNCATED()] * 20, max_tokens=2048)
        self.assertLessEqual(max(self.caps), llm_client.TRUNCATION_RETRY_CEILING)

    def test_truncation_has_its_own_attempt_budget(self):
        # Escalating the cap alone did not stop the failures in practice, so truncation
        # gets more attempts than the general retry count allows.
        n = llm_client.TRUNCATION_MAX_ATTEMPTS
        text, _ = self.call([TRUNCATED()] * (n - 1) + [response("Yes")],
                            max_tokens=256, max_retries=2)
        self.assertEqual(text, "Yes")
        self.assertEqual(len(self.caps), n)

    def test_persistent_truncation_still_fails_loudly(self):
        # The pipeline must never silently score a lost pair as a non-match.
        with self.assertRaises(RuntimeError):
            self.call([TRUNCATED()] * llm_client.TRUNCATION_MAX_ATTEMPTS, max_tokens=256)

    def test_truncation_budget_does_not_extend_the_empty_body_budget(self):
        with self.assertRaises(RuntimeError):
            self.call([response("")] * 4, max_tokens=256, max_retries=2)
        self.assertEqual(len(self.caps), 3, "max_retries=2 means three attempts")

    def test_first_attempt_uses_the_requested_cap(self):
        self.call([response("Yes")], max_tokens=8)
        self.assertEqual(self.caps, [8], "an unproblematic call must not over-reserve")

    def test_empty_content_retries_without_escalating(self):
        # A blank body is a different failure: the cap was not the problem.
        text, _ = self.call([response(""), response("Yes")], max_tokens=256)
        self.assertEqual(text, "Yes")
        self.assertEqual(self.caps, [256, 256])

    def test_default_empty_budget_recovers_after_three_blank_responses(self):
        text, usage = self.call([response("")] * 3 + [response("Yes")], max_tokens=256)
        self.assertEqual(text, "Yes")
        self.assertEqual(self.caps, [256] * 4)
        self.assertEqual(usage["total_tokens"], 48)
        self.assertEqual(len(usage["attempts"]), 4)
        self.assertEqual(sum("error" in a for a in usage["attempts"]), 3)

    def test_persistent_empty_responses_preserve_usage_and_fail_after_six_attempts(self):
        with self.assertRaises(llm_client.CompletionError) as ctx:
            self.call([response("")] * 10)
        self.assertEqual(len(self.caps), 6)
        self.assertEqual(ctx.exception.usage["total_tokens"], 72)
        self.assertEqual(len(ctx.exception.usage["attempts"]), 6)

    def test_retry_history_retains_provider_and_raw_response(self):
        empty = response("")
        empty.provider = "first"
        success = response("No")
        success.provider = "second"
        text, usage = self.call([empty, success])
        self.assertEqual(usage["provider"], "second")
        self.assertEqual([a["provider"] for a in usage["attempts"]], ["first", "second"])
        self.assertEqual(usage["attempts"][-1]["raw_response"], "No")
        self.assertEqual(usage["total_tokens"], 24)


if __name__ == "__main__":
    unittest.main()
