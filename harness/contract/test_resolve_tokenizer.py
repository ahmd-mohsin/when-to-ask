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


def test_auto_reads_SHARDED_manifests(tmp_path):
    """collect_v2 --num-shards N suffixes the manifest per shard, so the
    unsuffixed name never exists on a sharded collection. Before this was
    globbed, 'auto' silently returned the 7B default and mislabeled the
    Qwen3-32B R1 pilot (2026-08-08)."""
    from wta.labeling import resolve_tokenizer

    for shard in (0, 1):
        (tmp_path / f"collection_manifest.s{shard}.json").write_text(
            json.dumps({"args": {"model_id": "Qwen/Qwen3-32B"}}),
            encoding="utf-8")
    assert resolve_tokenizer(tmp_path) == "Qwen/Qwen3-32B"


def test_disagreeing_manifests_raise(tmp_path):
    """One tokenizer cannot label a collection built by two backbones."""
    import pytest

    from wta.labeling import resolve_tokenizer

    (tmp_path / "collection_manifest.s0.json").write_text(
        json.dumps({"args": {"model_id": "Qwen/Qwen3-32B"}}), encoding="utf-8")
    (tmp_path / "collection_manifest.s1.json").write_text(
        json.dumps({"args": {"model_id": "Qwen/Qwen2.5-Coder-7B-Instruct"}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="disagree on model_id"):
        resolve_tokenizer(tmp_path)
