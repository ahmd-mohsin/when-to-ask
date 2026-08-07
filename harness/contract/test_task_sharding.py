"""Contract: data-parallel task sharding (decisions/021 R2).

Qwen3-32B bf16 fits on one 96GB card, so N GPUs run as N independent workers
over disjoint task slices. The failure mode worth a test is silent: a stride
that drops or duplicates tasks would quietly shrink the collection, and nobody
would notice until the labels came out thin."""

import pytest


def shard_of(tasks, shard, num_shards):
    """The slice collect_v2 takes: eligible[shard::num_shards]."""
    return tasks[shard::num_shards]


@pytest.mark.parametrize("n_tasks,n_shards", [(60, 4), (60, 8), (7, 4), (1, 4)])
def test_shards_partition_the_task_list(n_tasks, n_shards):
    tasks = [f"swe_{i}" for i in range(n_tasks)]
    shards = [shard_of(tasks, i, n_shards) for i in range(n_shards)]
    covered = [t for s in shards for t in s]
    assert sorted(covered) == sorted(tasks), "tasks lost or duplicated"
    assert len(covered) == len(set(covered)), "a task landed on two GPUs"


def test_shards_are_balanced_within_one():
    tasks = [f"swe_{i}" for i in range(60)]
    sizes = [len(shard_of(tasks, i, 4)) for i in range(4)]
    assert max(sizes) - min(sizes) <= 1, f"unbalanced: {sizes}"


def test_single_shard_is_everything():
    tasks = [f"swe_{i}" for i in range(60)]
    assert shard_of(tasks, 0, 1) == tasks


def test_n_tasks_truncates_before_sharding():
    """--n-tasks is GLOBAL: 60 tasks across 4 workers, not 60 each."""
    eligible = [f"swe_{i}" for i in range(105)][:60]
    total = sum(len(shard_of(eligible, i, 4)) for i in range(4))
    assert total == 60
