# Clone (or refresh) the vendored upstream repos into third_party/ at pinned commits.
# We migrate their core algorithms into src/xtid/ but never edit the clones.
# Usage:  pwsh scripts/clone_third_party.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tp = Join-Path $root "third_party"
New-Item -ItemType Directory -Force -Path $tp | Out-Null

# name | url | pinned commit (see third_party/VERSIONS.md)
$repos = @(
    @{ name = "hil-bench";      url = "https://github.com/hilbenchauthors/hil-bench.git"; commit = "352d14c" },
    @{ name = "ClarifyGPT";     url = "https://github.com/ClarifyGPT/ClarifyGPT.git";     commit = "543b34b" },
    @{ name = "eigenscore";     url = "https://github.com/D2I-ai/eigenscore.git";         commit = "ea8062a" },
    @{ name = "OPENIA";         url = "https://github.com/iSE-UET-VNU/OPENIA.git";        commit = "aa96070" },
    @{ name = "mini-swe-agent"; url = "https://github.com/SWE-agent/mini-swe-agent.git";  commit = "531dbaf" }
    # STARS (https://github.com/lythk88/STARS) was empty as of 2026-06-12; the Stiefel
    # volume metric is reimplemented in src/xtid/signals/internal_divergence.py.
)

foreach ($r in $repos) {
    $dest = Join-Path $tp $r.name
    if (Test-Path (Join-Path $dest ".git")) {
        Write-Host "[skip] $($r.name) already cloned"
        continue
    }
    Write-Host "[clone] $($r.name) <- $($r.url)"
    git clone $r.url $dest
    if ($r.commit) {
        git -C $dest checkout $r.commit 2>$null
        if (-not $?) { Write-Host "  (warning: could not pin $($r.name) to $($r.commit); using default branch)" }
    }
}
Write-Host "Done. See third_party/VERSIONS.md for what we migrate from each."
