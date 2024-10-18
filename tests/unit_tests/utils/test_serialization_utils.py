import pytest
from datetime import datetime
from guardrails.utils.serialization_utils import serialize, deserialize


class TestSerializeAndDeserialize:
    def test_string(self):
        data = "value"

        serialized_data = serialize(data)
        assert serialized_data == '"value"'

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data == data

    def test_int(self):
        data = 1

        serialized_data = serialize(data)
        assert serialized_data == "1"

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data == data

    def test_float(self):
        data = 1.0

        serialized_data = serialize(data)
        assert serialized_data == "1.0"

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data == data

    def test_bool(self):
        data = True

        serialized_data = serialize(data)
        assert serialized_data == "true"

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data == data

    def test_datetime(self):
        data = datetime(2024, 9, 10, 0, 0, 0)

        serialized_data = serialize(data)
        assert serialized_data == '"2024-09-10T00:00:00"'

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data == data

    def test_dictionary(self):
        data = {"key": "value"}

        serialized_data = serialize(data)
        assert serialized_data == '{"key": "value"}'

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data == data

    def test_list(self):
        data = ["value1", "value2"]

        serialized_data = serialize(data)
        assert serialized_data == '["value1", "value2"]'

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data == data

    def test_simple_class(self):
        class TestClass:
            def __init__(self, key: str):
                self.key = key

        data = TestClass("value")

        serialized_data = serialize(data)
        assert serialized_data == '{"key": "value"}'

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data.key == data.key

    def test_nested_classes_not_supported(self):
        class TestClass:
            def __init__(self, value: str):
                self.value = value

        class TestClass2:
            def __init__(self, value: TestClass):
                self.value = value

        data = TestClass2(TestClass("value"))

        serialized_data = serialize(data)
        assert serialized_data == '{"value": {"value": "value"}}'

        deserialized_data = deserialize(data, serialized_data)
        with pytest.raises(AttributeError) as excinfo:
            assert deserialized_data.value.value == data.value.value

        assert str(excinfo.value) == "'dict' object has no attribute 'value'"

    def test_simple_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            key: str

        data = TestClass("value")

        serialized_data = serialize(data)
        assert serialized_data == '{"key": "value"}'

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data.key == data.key

    def test_nested_dataclasses_not_supported(self):
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            value: str

        @dataclass
        class TestClass2:
            value: TestClass

        data = TestClass2(TestClass("value"))

        serialized_data = serialize(data)
        assert serialized_data == '{"value": {"value": "value"}}'

        deserialized_data = deserialize(data, serialized_data)
        with pytest.raises(AttributeError) as excinfo:
            assert deserialized_data.value.value == data.value.value

        assert str(excinfo.value) == "'dict' object has no attribute 'value'"

    def test_simple_pydantic_model(self):
        from pydantic import BaseModel

        class TestClass(BaseModel):
            key: str

        data = TestClass(key="value")

        serialized_data = serialize(data)
        assert serialized_data == '{"key": "value"}'

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data.key == data.key

    def test_nested_pydantic_models(self):
        from pydantic import BaseModel

        class TestClass(BaseModel):
            value: str

        class TestClass2(BaseModel):
            value: TestClass

        data = TestClass2(value=TestClass(value="value"))

        serialized_data = serialize(data)
        assert serialized_data == '{"value": {"value": "value"}}'

        deserialized_data = deserialize(data, serialized_data)
        assert deserialized_data.value.value == data.value.value
