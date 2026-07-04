# Spec B — Online trigger (Part B)

Built under decisions/011: implementation now, but **no Part B result is
trusted or reported until the A4 gates pass on real held-out data and the
owner reviews them**.

## Inputs per read (all frozen forward-pass products; nothing else)

`observe(run_id, topic_vec=T(h), r_vec=L(h), s, weight)` where `weight` is
A3's soft commitment weight (0 = not committed). **The API takes no step
index and no action** — actions enter only via `register_action(run_id,
text)`, kept solely to phrase the fired question's options. Environment-state
signatures enter via `notify_env_state(run_id, hash)` for the loop channel.

## Bucketing (wta/bucketing.py)

Leader clustering on the topic stream, cosine threshold `theta` (set from A4
gate 3's same-vs-different crossover):

- nearest bucket ≥ theta joins and updates the unit-norm running-mean
  centroid; else a new bucket is seeded;
- **hysteresis**: a run stays in its bucket unless its topic leaves theta for
  M consecutive reads (out-of-theta reads never update the centroid);
- **merge**: any two centroids within `merge_theta` collapse (to fixpoint);
  votes pool, a run's most recent vote (seq stamp) wins; makes the final
  bucketing arrival-order-invariant in practice.

## Voting — mutable

Commit (`weight > 0`) ⇒ `votes[run] = (r, weight, seq)` (overwrite = the
run's CURRENT committed lean). De-commit ⇒ retract. A slow run's vote lands
in the right bucket whenever it arrives (async dissolves by construction).

## Trigger — CUSUM over commitment-weighted dispersion + loop channel

```
spread_b = weighted mean pairwise ||r_i − r_j|| over the bucket's votes
           (weights w_i·w_j; soft — never hard-exclude a run; 0 if < min_votes)
D_loop_b = Σ_{runs assigned to b} max(0, max env-state repeats − floor + 1)
D_total  = spread_b + lambda_loop · D_loop_b
S_b      = max(0, S_b + D_total − reference − slack)      # online CUSUM
fire when S_b > h_threshold
```

Fires only on **persistent** dispersion: a transient blip contributes a few
`spread − reference − slack` increments and decays back to 0 once mutable
votes re-converge; a genuine fork keeps pumping. Note the CUSUM's time unit
is **read-events on the bucket** — every observe of any assigned run pumps
it, so one "round" of persistent spread contributes ~N increments at N runs;
`h_threshold` is sized in those units and is a Phase-2 sweep alongside the
per-bucket normalization open item. A looping run (repeated env
states, never commits, never votes) still counts through `D_loop`.

On fire: options = the divergent votes (r vectors + their runs' last action
texts, sorted by weight) plus the list of looping runs; the caller asks the
human, then `inject_resolution(bucket_id)` clears votes + pressure and all
runs continue.

## Observable behaviour that verifies this spec (contract)

1. Persistent two-vote disagreement fires within a bounded number of reads;
   the decision carries both options.
2. A transient blip (2 reads of disagreement, then votes re-converge) does
   NOT fire with default parameters, and pressure decays back to 0.
3. De-commitment retracts the vote (spread returns to 0) — wired to A3's
   de-commit behaviour.
4. Hysteresis: topic out-of-theta for < M reads does not move the run or
   pollute the centroid; M consecutive reads does.
5. Merge: permuted arrival orders of the same reads end with the same bucket
   count and memberships.
6. Loop channel: a run with repeated env states and weight 0 throughout
   drives a fire via `lambda_loop`; it appears in `looping_runs`.
7. The observe path never receives an action or step index (API shape).
