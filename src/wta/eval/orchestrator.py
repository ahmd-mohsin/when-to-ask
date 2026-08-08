"""Lockstep N-run orchestrator for one (arm, task, pass) (spec eval,
decisions/022).

Round-robin: one ``TurnStepper.step()`` per live run per round, all runs
sharing ONE model session (the in-process HFStreamReader at eval time; fakes
in tests) and each owning its own env (DockerTaskEnv on GPU; fakes in tests).
After each full round the policy is drained: every AskRequest goes through
the ONE judge path, is recorded in the AskSession (per-blocker cap enforced
there), and the response is injected into ALL runs as a user turn
(decisions/022 §2c). Model-initiated asks (naive-ask arm) surface mid-round
via StepOutcome.asked and are answered as that run's command observation --
hil-bench's own ask_human tool semantics.

Pass outcome for N-run arms is the pre-registered committed-trajectory rule
(decisions/022 §2a): lowest-seed finished run, else lowest-seed run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from wta.agent_loop import AgentLoopConfig
from wta.eval.fullinfo import augment_full_info
from wta.eval.patch import extract_patch, prediction_row
from wta.eval.policies import TaskContext, TriggerPolicy
from wta.eval.stepper import TurnStepper

_ANSWER_TEMPLATE = "[HUMAN ANSWER] Question: {q}\nAnswer: {a}"


@dataclass
class ArmSpec:
    name: str
    n_runs: int = 1
    ask_affordance: bool = False       # model_initiated arm
    full_info: bool = False
    policy_factory: object | None = None   # () -> TriggerPolicy
    temperatures: tuple = (1.0,)       # cycled over runs (decisions/022 §2d)


@dataclass
class TaskPassResult:
    arm: str
    task_id: str
    pass_idx: int
    instance_key: str
    runs: dict = field(default_factory=dict)        # run_id -> AgentRunResult
    seeds: dict = field(default_factory=dict)       # run_id -> seed
    ask_events: list = field(default_factory=list)
    policy_stats: dict = field(default_factory=dict)
    committed_run_id: str | None = None
    prediction: dict | None = None                  # preds.json row
    compute: dict = field(default_factory=dict)


def committed_run(results: dict, seeds: dict) -> str:
    """decisions/022 §2a: lowest-seed run that finished; else lowest-seed run."""
    ordered = sorted(results, key=lambda rid: seeds[rid])
    for rid in ordered:
        if results[rid].finished:
            return rid
    return ordered[0]


def run_task_pass(arm: ArmSpec, task, *, session, env_factory, judge,
                  ask_session, cfg: AgentLoopConfig, pass_idx: int,
                  seed_base: int, model_id: str = "", mid_layer: int = 0,
                  layers=None, extract_patches: bool = True) -> TaskPassResult:
    """`task` is an xtid Task (instance_id, statement, blockers).
    `env_factory(run_id) -> env` (fresh container per run on GPU)."""
    if task.meta.get("statement_stub"):
        raise ValueError(f"{task.instance_id}: stub statement -- refusing to eval")

    instruction = task.statement
    if arm.full_info:
        instruction = augment_full_info(instruction, task.blockers)

    instance_key = f"{task.instance_id}__{arm.name}__pass_{pass_idx}"
    ask_session.start_instance(instance_key, [b.id for b in task.blockers])

    result = TaskPassResult(arm=arm.name, task_id=task.instance_id,
                            pass_idx=pass_idx, instance_key=instance_key)

    steppers, envs = {}, {}
    for i in range(arm.n_runs):
        seed = seed_base + i
        run_id = f"{task.instance_id}-p{pass_idx}-s{seed}"
        run_cfg = replace(cfg, temperature=arm.temperatures[i % len(arm.temperatures)])
        env = env_factory(run_id)
        envs[run_id] = env
        steppers[run_id] = TurnStepper(
            session, env, instruction, run_id=run_id, task_id=task.instance_id,
            seed=seed, cfg=run_cfg, ask_affordance=arm.ask_affordance,
            model_id=model_id, mid_layer=mid_layer, layers=layers)
        result.seeds[run_id] = seed

    policy: TriggerPolicy = (arm.policy_factory() if arm.policy_factory
                             else TriggerPolicy())
    policy.bind(TaskContext(task_id=task.instance_id, statement=task.statement,
                            steppers=steppers, session=session, seed=seed_base))

    n_phrasing_calls = 0
    round_idx = 0
    while any(not st.done() for st in steppers.values()):
        round_idx += 1
        for run_id, st in steppers.items():
            if st.done():
                continue
            out = st.step()
            policy.on_reads(run_id, out.reads)
            policy.on_turn(run_id, out)
            if out.asked is not None:
                jr = judge.ask(out.asked, task.blockers)
                response = ask_session.record(instance_key, out.asked, jr)
                st.answer_ask(response)
                result.ask_events.append({
                    "round": round_idx, "source": "model", "run_id": run_id,
                    "question": out.asked, "response": response,
                    "blocker_id": jr.blocker_id})

        while (req := policy.poll()) is not None:
            jr = judge.ask(req.question, task.blockers)
            response = ask_session.record(instance_key, req.question, jr)
            answer_msg = _ANSWER_TEMPLATE.format(q=req.question, a=response)
            for st in steppers.values():          # all N runs (022 §2c)
                if not st.done():
                    st.inject_user_message(answer_msg)
            policy.on_answer(req, jr)
            if req.meta.get("method") == "llm":
                n_phrasing_calls += 1
            result.ask_events.append({
                "round": round_idx, "source": policy.name, "run_id": None,
                "question": req.question, "response": response,
                "blocker_id": jr.blocker_id, "key": req.key,
                "meta": {k: v for k, v in req.meta.items() if k != "options"}})

    for run_id, st in steppers.items():
        result.runs[run_id] = st.result()

    result.policy_stats = dict(policy.stats)
    result.committed_run_id = committed_run(result.runs, result.seeds)
    if extract_patches:
        patch = extract_patch(envs[result.committed_run_id])
        result.prediction = prediction_row(task.instance_id, patch,
                                           model_name=model_id or arm.name)
    result.compute = {
        "n_runs": arm.n_runs,
        "rounds": round_idx,
        "turns": sum(r.n_steps for r in result.runs.values()),
        "generated_chars": sum(len(s) for r in result.runs.values()
                               for s in r.segments),
        "asks": len(result.ask_events),
        "phrasing_calls": n_phrasing_calls,
    }
    return result
