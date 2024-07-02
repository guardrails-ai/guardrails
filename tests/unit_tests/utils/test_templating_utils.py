from guardrails.utils.templating_utils import get_template_variables


def test_get_template_variables():
    string_template = "${my_var} $my_second_var {not_a_var}"
    vars = get_template_variables(string_template)

    assert vars == ["my_var", "my_second_var"]
