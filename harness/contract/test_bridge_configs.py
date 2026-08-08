"""Contract: bridge config invariants (spec eval-bridge, decisions/022).

Parses our configs/hilbench/* and pins every invariant the bridge depends on:
self_hosted hosting via AGENT_SWE_BASE_URL, thought_action + bash pinned,
sampling pinned to the single-trajectory protocol (temp 1.0 / top_p 1.0),
ask config's container->host plumbing, judge config byte-equivalence to the
vendored frozen one, and mapping paths that resolve from the harness CWD.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "configs" / "hilbench"
HIL = REPO / "third_party" / "hil-bench"


def _load(name):
    return yaml.safe_load((CFG / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["swe_default_qwen3_32b.yaml",
                                  "swe_ask_qwen3_32b.yaml"])
def test_agent_config_invariants(name):
    cfg = _load(name)
    assert cfg["hosting"] == {"type": "self_hosted",
                              "api_base_env": "AGENT_SWE_BASE_URL"}
    tools = cfg["agent"]["tools"]
    assert tools["parse_function"] == {"type": "thought_action"}
    assert tools["enable_bash_tool"] is True
    model = cfg["agent"]["model"]
    assert model["temperature"] == 1.0 and model["top_p"] == 1.0  # 022 §2d
    assert "api_base" not in model          # injected from env, never inline
    bundle_paths = [b["path"] for b in tools["bundles"]]
    assert "tools/registry" in bundle_paths


def test_ask_config_container_to_host_plumbing():
    cfg = _load("swe_ask_qwen3_32b.yaml")
    tools = cfg["agent"]["tools"]
    bundle_paths = [b["path"] for b in tools["bundles"]]
    assert "tools/ask_human" in bundle_paths
    prop = tools["propagate_env_variables"]
    assert "ASK_HUMAN_SERVER_URL" in prop and "ASK_HUMAN_MODEL" in prop
    assert "TASK_INSTANCE_ID" not in prop   # upstream: post_startup sets it
    docker_args = cfg["instances"]["deployment"]["docker_args"]
    assert "--add-host=host.docker.internal:host-gateway" in docker_args
    assert "ask_human" in cfg["agent"]["templates"]["system_template"]


def test_default_config_has_no_ask_surface():
    cfg = _load("swe_default_qwen3_32b.yaml")
    bundle_paths = [b["path"] for b in cfg["agent"]["tools"]["bundles"]]
    assert "tools/ask_human" not in bundle_paths
    assert "ask_human" not in cfg["agent"]["templates"]["system_template"]


def test_judge_config_matches_vendored_frozen_one():
    ours = _load("judge_config.yaml")
    theirs = yaml.safe_load((HIL / "judge_config.yaml").read_text(encoding="utf-8")) \
        if (HIL / "judge_config.yaml").exists() else None
    assert ours["model"] == "casperhansen/llama-3.3-70b-instruct-awq"
    assert ours["hosting"]["type"] == "self_hosted"
    assert ours["hosting"]["self_hosted_base_url"] == "http://127.0.0.1:8808"
    if theirs is not None:
        assert ours == theirs               # byte-equivalent semantics


def test_mapping_paths_resolve_from_harness_cwd():
    mapping = yaml.safe_load((CFG / "config_mappings.yaml").read_text(encoding="utf-8"))
    swe = mapping["swe"]
    assert set(swe) == {"baseline", "ask_human", "full_info"}
    model_keys = {k for mode in swe.values() for k in mode}
    assert len(model_keys) == 1             # one backbone, consistent key
    for mode, entries in swe.items():
        for path in entries.values():
            # resolved against the harness CWD (third_party/hil-bench)
            assert (HIL / path).resolve().exists(), f"{mode}: {path}"
    # baseline and full_info share the no-ask config; ask_human differs
    assert swe["baseline"] == swe["full_info"] != swe["ask_human"]
