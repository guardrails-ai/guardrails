from guardrails.utils.regex_utils import split_on, ESCAPED_OR_QUOTED


class TestSplitOn:
    def test_happy_path(self):
        string = 'length: 0 1; ends-with: {"some text;"};'
        tokens = split_on(string, ";")
        assert len(tokens) == 2
        assert tokens == ["length: 0 1", 'ends-with: {"some text;"}']

    def ignore_test_quoted(self):
        string = "length: 0 1; ends-with: {\"some text;\"}; other: 'don't escape; this';"  # noqa
        tokens = split_on(string, ";", exceptions=ESCAPED_OR_QUOTED)
        assert len(tokens) == 3
        assert tokens == [
            "length: 0 1",
            'ends-with: {"some text;"}',
            "other: 'don't escape this;'",
        ]

    def test_no_filter(self):
        string = 'length: 0 1; ends-with: {"some text;"};'
        tokens = split_on(string, ";", filter_nones=False)
        assert len(tokens) == 3
        assert tokens == ["length: 0 1", 'ends-with: {"some text;"}', ""]
