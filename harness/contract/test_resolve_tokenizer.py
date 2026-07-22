"""Contract: tokenizer resolution for labeling (no run data needed).

Token->char maps are built by re-tokenizing the trace, so the labeler must
use the COLLECTION model's tokenizer — 'auto' reads it from the collection
manifest (the Qwen3-32B collection must not be labeled with the Qwen2.5
default)."""

import json


def test_auto_reads_manifest_model_id(tmp_path):
    from wta.labeling import resolve_tokenizer

    (tmp_path / "collection_manifest.json").write_text(
        json.dumps({"args": {"model_id": "Qwen/Qwen3-32B"}}), encoding="utf-8")
    assert resolve_tokenizer(tmp_path) == "Qwen/Qwen3-32B"


def test_auto_without_manifest_falls_back(tmp_path):
    from wta.labeling import resolve_tokenizer

    assert resolve_tokenizer(tmp_path) == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_explicit_name_passes_through(tmp_path):
    from wta.labeling import resolve_tokenizer

    (tmp_path / "collection_manifest.json").write_text(
        json.dumps({"args": {"model_id": "Qwen/Qwen3-32B"}}), encoding="utf-8")
    assert resolve_tokenizer(tmp_path, "some/other-model") == "some/other-model"
