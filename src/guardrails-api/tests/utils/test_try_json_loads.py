from guardrails_api.utils.try_json_loads import try_json_loads


class TestTryJsonLoads:
    def test_pass(self):
        val = '{"a": 1}'

        actual = try_json_loads(val)

        assert actual == {"a": 1}

    def test_fail(self):
        val = "not a json object"

        actual = try_json_loads(val)

        assert actual == "not a json object"
