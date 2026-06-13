#!/usr/bin/env bash
# Clone (or refresh) the vendored upstream repos into third_party/ at pinned commits.
# We migrate their core algorithms into src/xtid/ but never edit the clones.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

# name url commit  (see third_party/VERSIONS.md)
clone() {
  local name="$1" url="$2" commit="$3" dest="third_party/$1"
  if [ -d "$dest/.git" ]; then echo "[skip] $name already cloned"; return; fi
  echo "[clone] $name <- $url"
  git clone "$url" "$dest"
  if [ -n "$commit" ]; then git -C "$dest" checkout "$commit" 2>/dev/null \
    || echo "  (warning: could not pin $name to $commit; using default branch)"; fi
}

clone hil-bench      https://github.com/hilbenchauthors/hil-bench.git 352d14c
clone ClarifyGPT     https://github.com/ClarifyGPT/ClarifyGPT.git     543b34b
clone eigenscore     https://github.com/D2I-ai/eigenscore.git         ea8062a
clone OPENIA         https://github.com/iSE-UET-VNU/OPENIA.git        aa96070
clone mini-swe-agent https://github.com/SWE-agent/mini-swe-agent.git  531dbaf
# STARS (https://github.com/lythk88/STARS) was empty as of 2026-06-12; the Stiefel
# volume metric is reimplemented in src/xtid/signals/internal_divergence.py.
echo "Done. See third_party/VERSIONS.md."
