# Provenance — mini-swe-agent

- **Repo:** https://github.com/SWE-agent/mini-swe-agent
- **Commit:** 531dbaf (cloned 2026-06-12, xtid pass)
- **Licence:** MIT (© 2025 Kilian A. Lieret and Carlos E. Jimenez)
- **What we migrated (xtid pass):** the minimal observe→query→act agent loop
  (`src/minisweagent/agents/default.py`) → `src/xtid/agent/loop.py`. The wta
  N-trajectory runner builds on that port (decisions/001).
