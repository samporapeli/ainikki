import json

from agent.json_utils import strip_code_fences


def test_clean_json_passes_through():
    raw = '{"key": "value", "number": 42}'
    assert strip_code_fences(raw) == raw


def test_json_with_language_annotation():
    raw = '```json\n{"key": "value"}\n```'
    result = strip_code_fences(raw)
    assert result == '{"key": "value"}'
    parsed = json.loads(result)
    assert parsed["key"] == "value"


def test_json_with_bare_fences():
    raw = '```\n{"key": "value"}\n```'
    result = strip_code_fences(raw)
    assert result == '{"key": "value"}'
    json.loads(result)  # ei saa heittää poikkeusta


def test_json_with_whitespace_padding():
    raw = '  ```json\n  {"key": "value"}  \n```  '
    result = strip_code_fences(raw)
    assert result == '{"key": "value"}'
    json.loads(result)


def test_empty_string():
    assert strip_code_fences("") == ""


def test_whitespace_only():
    assert strip_code_fences("   \n  \n  ") == ""


def test_nested_json():
    inner = {"clusters": [{"indices": [0, 1]}]}
    raw = f"```json\n{json.dumps(inner)}\n```"
    result = strip_code_fences(raw)
    parsed = json.loads(result)
    assert parsed == inner

