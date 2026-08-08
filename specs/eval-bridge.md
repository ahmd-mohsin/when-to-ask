# Spec — Server bridge (bridge rows in hil-bench's own harness)

New component spec (decisions/019 tier-3, decisions/021 §8, decisions/022).
Purpose: the arms that need no activation reads — **no-ask (`baseline`),
naive-ask (`ask_human`), full-info (`full_info`)** — are ALSO run inside
hil-bench's own harness (SWE-Agent scaffold) with our backbone self-hosted
behind vLLM. If harness-row ≈ our-loop-row for the same arm, the scaffold delta
is measured, not argued. The detector arm NEVER runs through the bridge
(activation capture requires the model in-process).

## Inputs / outputs

- In: the sealed task pool as a **flat tasks dir** (see materialization), our
  agent/judge configs in `configs/hilbench/`, a vLLM endpoint for the backbone
  (`AGENT_SWE_BASE_URL`) and one for the judge (:8808).
- Out (per upstream): per-pass SWE-Agent trajectories + `preds.json`,
  `ask_human_logs.json` (ask mode), from which OUR scoring recomputes Ask-F1
  and applies the ported pass@3 semantics (`src/xtid/harness/passk.py`).

## Invariants (contract-tested where testable on CPU)

1. **third_party/ is byte-untouched.** All configuration enters via the
   harness CLI: `--config-mapping configs/hilbench/config_mappings.yaml`,
   `--judge-config configs/hilbench/judge_config.yaml`.
2. **Agent hosting**: `hosting: {type: self_hosted, api_base_env:
   AGENT_SWE_BASE_URL}` in our agent YAMLs. No `LITELLM_BASE_URL` set.
3. **Parser pinned**: `parse_function: {type: thought_action}` +
   `enable_bash_tool: true` — with a custom `api_base` the harness SKIPS its
   function-calling capability downgrade (swe.py:647-672), so leaving
   `function_calling` would silently require OpenAI tool-calling from the
   served model.
4. **Cost limit**: a model name unknown to `litellm.model_cost` gets
   `per_instance_cost_limit` forced to 0 — verified in swe.py this means
   DISABLED ("disable cost limit to avoid ModelConfigurationError"), which is
   correct for a self-hosted backbone: run length is bounded by `--max-steps`,
   not cost. No litellm registry file is needed (the registry lookup only
   re-enables cost *tracking*, and it globs the hil-bench repo's own configs/,
   which we do not touch).
5. **Sampling pinned** (decisions/022 §2d): temperature 1.0, top_p 1.0 in the
   agent YAML — identical to the single-trajectory our-loop arms, so bridge vs
   our-loop differs in scaffold (and engine, disclosed) only.
6. **Judge config byte-equivalent** to the vendored `judge_config.yaml`
   (model `casperhansen/llama-3.3-70b-instruct-awq`, `self_hosted`,
   `http://127.0.0.1:8808`).
7. **Container→host reachability**: `docker_args` keep `--entrypoint=` and
   `--add-host=host.docker.internal:host-gateway`; `propagate_env_variables`
   keep `ASK_HUMAN_MODEL` / `ASK_HUMAN_SERVER_URL`; `TASK_INSTANCE_ID` is NOT
   propagated (upstream comment: post_startup_commands set it per-instance).
8. **Ask-F1 recompute rule** (decisions/022 §3): bridge Ask-F1 is computed by
   our `compute_hil_metrics` from `ask_human_logs.json` + true registry
   blocker counts. Upstream's summary is not trusted for zero-question passes
   (`batch_runner.py:1179` imports a nonexistent `compute_zero_hil_metrics`;
   the swallowed ImportError zeroes the recall denominator) and hides ask
   metrics for any mode not literally named `ask_human`.
9. **Sealed-pool selection** goes through `python -m hil_bench.cli swe
   <flat-dir>` over our materialized dir — `run_hil_bench.py` re-downloads the
   HF dataset and can only take the first-N datapoints.

## Materialization contract (`scripts/materialize_hilbench_tasks.py`)

For each `harbor_swe/swe_i/`: copy `shared/problem_statement.txt`,
`shared/metadata.json`, `shared/ask-human-data/blocker_registry.json` to the
flat task-dir **root** (the layout `resolve_swe_input_path` +
`ask_human_server.main --tasks-dir` require). `--extract-scripts` (GPU/docker
only) additionally docker-cats `/root/run_script.sh` + `/root/parser.py` from
the image named in `metadata.json` (image pulled via the verified
`extract_task_context.try_load_archive` path) so `calculate_pass_at_1` can
score. The materializer never touches the source tree and refuses to mix train
and sealed pools in one output dir.

## Scaffold-robustness check (decisions/021 §8 item 4)

`scripts/scaffold_robustness.py` parses the bridge's SWE-Agent `.traj` files,
runs OUR composite-observable extractors (`wta.agent_loop`) over each step's
command+thought, and compares cross-pass fork-signature statistics against the
same statistics from our-loop no-ask trajectories on the same tasks. Claim
tested: fork structure is not an artifact of the one-bash-block scaffold.
**Behavioral-only** — bridge runs go through vLLM, so no activations exist on
that side; stated as a scope limit, not hidden.

## Run procedure

GPU-time only, AWS_RUNBOOK step 5. Bridge rows on the sealed pool run ONCE
(seal rule, decisions/018 §4), after the judge-validation step passes and
after a 1-task train-pool smoke.
