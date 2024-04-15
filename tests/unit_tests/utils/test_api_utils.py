from guardrails.utils.api_utils import extract_serializeable_metadata


def test_extract_serializeable_metadata():
    def baz():
        print("baz")

    class NonMeta:
        data = "data"

    metadata = {
        "foo": "bar",
        "baz": baz,
        "non_meta": NonMeta(),
    }
    
    extracted_metadata = extract_serializeable_metadata(metadata)
    
    assert extracted_metadata == {
        "foo": "bar"
    }