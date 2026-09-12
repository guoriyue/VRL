import json

import pytest

from vrl.utils.json_files import read_json, read_jsonl, write_json, write_jsonl


def test_write_json_creates_parents_and_sorts_keys(tmp_path):
    path = tmp_path / "nested" / "report.json"

    write_json(path, {"b": 1, "a": [1, 2]})

    assert path.read_text(encoding="utf-8") == '{\n  "a": [\n    1,\n    2\n  ],\n  "b": 1\n}\n'
    assert read_json(path) == {"a": [1, 2], "b": 1}
    assert sorted(tmp_path.joinpath("nested").iterdir()) == [path]


def test_write_json_without_overwrite_keeps_the_first_record(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"attempt": 1}, overwrite=False)

    with pytest.raises(FileExistsError):
        write_json(path, {"attempt": 2}, overwrite=False)

    assert read_json(path) == {"attempt": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_replaces_an_existing_file_by_default(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"attempt": 1})
    write_json(path, {"attempt": 2})

    assert read_json(path) == {"attempt": 2}


def test_jsonl_round_trip_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"b": 1, "a": "x"}, {"a": "y"}]

    assert write_jsonl(path, rows) == 2
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    assert read_jsonl(path) == [{"a": "x", "b": 1}, {"a": "y"}]
    assert path.read_text(encoding="utf-8").splitlines()[0] == json.dumps(rows[0], sort_keys=True)


def test_read_jsonl_rejects_a_non_object_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2 must be a JSON object"):
        read_jsonl(path)


def test_failed_jsonl_iteration_preserves_existing_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"old": True}])
    original = path.read_bytes()

    def rows():
        yield {"new": True}
        raise RuntimeError("input iteration failed")

    with pytest.raises(RuntimeError, match="input iteration failed"):
        write_jsonl(path, rows())
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
