from unittest.mock import Mock
from pydantic import BaseModel

from clustering_pipeline.feature_extraction.feature_extraction import extract_features_from_image


class DummyModel:
    def __init__(self, **data):
        # accept anything
        pass


class StrictModel(BaseModel):
    name: str
    age: int



def test_extract_success(tmp_path):
    """
    Test that extract_features_from_image correctly converts a string response to JSON.
    """
    # create fake prompt file
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("test prompt")

    # Mock LLM
    llm = Mock()
    llm.generate_response.return_value = '{"a": 1, "b": 2}'

    result = extract_features_from_image(
        image_path="img.png",
        llm=llm,
        prompt_path=prompt_file,
        response_model=DummyModel,
        max_retries=3
    )

    assert result == {"a": 1, "b": 2}
    llm.generate_response.assert_called_once()


def test_invalid_json_then_success(tmp_path):
    """
    Test that extract_features_from_image retries on invalid JSON response.
    """
    # create fake prompt file
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt")

    # Mock LLM
    llm = Mock()
    llm.generate_response.side_effect = [
        "not json",
        '{"x": 42}'
    ]

    result = extract_features_from_image(
        "img.png",
        llm=llm,
        prompt_path=prompt_file,
        response_model=DummyModel,
        max_retries=2
    )

    assert result == {"x": 42}
    assert llm.generate_response.call_count == 2


def test_invalid_schema(tmp_path):
    """
    Test that extract_features_from_image retries on JSON that doesn't match the schema.
    """
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt")

    llm = Mock()
    llm.generate_response.side_effect = [
        '{"name": "John"}',  # missing age
        '{"name": "John", "age": 30}'
    ]

    result = extract_features_from_image(
        "img.png",
        llm=llm,
        prompt_path=prompt_file,
        response_model=StrictModel,
        max_retries=2
    )

    assert result == {"name": "John", "age": 30}
    assert llm.generate_response.call_count == 2


def test_all_failures(tmp_path):
    """
    Test that extract_features_from_image returns empty dict after all retries fail.
    """
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt")

    llm = Mock()
    llm.generate_response.return_value = "invalid json"

    result = extract_features_from_image(
        "img.png",
        llm=llm,
        prompt_path=prompt_file,
        response_model=DummyModel,
        max_retries=2
    )

    assert result == {}
    assert llm.generate_response.call_count == 3