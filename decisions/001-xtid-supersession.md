# 001 — xtid de-risk is superseded; new method lives in `src/wta/`

---
status: agreed
date: 2026-07-02
---

**Context.** The repo contains the earlier xtid de-risk pipeline (`src/xtid/`,
claim C1 from `pre-implementation brief.md`); its real-GPU run never happened.
The When-to-Ask build brief doesn't reference C1.

**Decision.** Owner: "sure" — the de-risk pass is superseded by this build.
`src/xtid/` is frozen as an artifact (not deleted, not extended). The
When-to-Ask method is built in a new package `src/wta/`, reusing xtid's
already-migrated components where they fit: HiL-Bench judge / Ask-F1 /
executor wrappers, `HFWhiteBoxModel` mid-layer hook, mini-swe-agent loop port.

**Consequences.** C1 does not gate this build. Anything reused from
`src/xtid/` is imported or copied with an in-file pointer, never edited
in place.
