import unittest
from guardrails.telemetry.common import ismatchingkey


class TestIsMatchingKey(unittest.TestCase):
    def test_key_matches_with_default_keys(self):
        self.assertTrue(ismatchingkey("api_key"))
        self.assertTrue(ismatchingkey("user_token"))
        self.assertFalse(ismatchingkey("username"))
        self.assertTrue(ismatchingkey("password"))

    def test_key_matches_with_custom_keys(self):
        self.assertTrue(ismatchingkey("api_secret", keys_to_match=("secret",)))
        self.assertTrue(ismatchingkey("client_id", keys_to_match=("id",)))
        self.assertFalse(ismatchingkey("session", keys_to_match=("key", "token")))

    def test_empty_key(self):
        self.assertFalse(ismatchingkey("", keys_to_match=("key", "token")))

    def test_empty_keys_to_match(self):
        self.assertFalse(ismatchingkey("key", keys_to_match=()))


if __name__ == "__main__":
    unittest.main()
