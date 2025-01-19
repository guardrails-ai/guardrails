import unittest
from guardrails.telemetry.common import redact


class TestRedactFunction(unittest.TestCase):
    def test_redact_long_string(self):
        self.assertEqual(redact("supersecretpassword"), "***************word")

    def test_redact_short_string(self):
        self.assertEqual(redact("test"), "test")

    def test_open_ai_example_key(self):
        self.assertEqual(
            redact("sk-1234abcdefghijklmnopqrstuvwxhp37"),
            "*******************************hp37",
        )

    def test_redact_very_short_string(self):
        self.assertEqual(redact("abc"), "abc")

    def test_redact_empty_string(self):
        self.assertEqual(redact(""), "")

    def test_redact_exact_length(self):
        self.assertEqual(redact("1234"), "1234")

    def test_redact_special_characters(self):
        self.assertEqual(redact("ab!@#12"), "***@#12")

    def test_redact_single_character(self):
        self.assertEqual(redact("a"), "a")

    def test_redact_spaces(self):
        self.assertEqual(redact("      test"), "******test")


if __name__ == "__main__":
    unittest.main()
