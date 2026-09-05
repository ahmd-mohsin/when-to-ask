# Working in this repo

**Read [STATUS.md](STATUS.md) first.** It decodes the code names (T#/F#, the
two different R# schemes, A0–A4, gate1–7) and carries the live done/left
board. It is the fastest orientation in the repo.

Then, as needed:
- [decisions/028-iclr-experimental-program.md](decisions/028-iclr-experimental-program.md)
  — the frozen experiment program currently being executed, plus Amendment A
  (as-implemented instantiation).
- [paper/OUTLINE.md](paper/OUTLINE.md) — the paper the results feed.
- `decisions/` — the pre-registration chain, newest last. Amendments record
  as-run outcomes including NO-GOs.

## Non-negotiable process rules

1. **STATUS.md is updated in the same commit as any experiment that lands or
   changes status.** Move the marker, paste the headline number and results
   path, update "Known gaps" if affected.
2. **Everything is pre-registered.** Numbers are reported **as-run**,
   whichever way they land. No rerun with variations without amending
   `decisions/028` FIRST.
3. Every experiment ships a script in `scripts/` and writes to a fresh file
   in `results/` — never overwrite prior artifacts.
4. The test suite stays green (currently 278 passed, 2 skipped). New frozen protocol choices
   get a contract test.
5. The **sealed pool (swe_60+) stays sealed.**
6. Commit each completed experiment with its results; push to BOTH
   `master:master` and `master:main`.
7. **Use the Bash tool for git** — PowerShell git stalls on this machine.
8. Scope boundary for this cycle: **measurement only, no mechanism claims.**
