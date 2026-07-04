"""Read-position policy: WHEN to read the mid-layer residual -- OURS (spec A0).

Reads happen throughout the reasoning span, at a fixed token cadence and at
deliberation cues, never only at the action boundary (decisions/006; the method
doc calls boundary-only reading "the single most common way to accidentally
kill the project"). Cue-token reading follows LYNX's precedent (cue tokens like
"hmm"/"wait"; third_party/LYNX/PROVENANCE.md); the cue set and the streaming
matcher are ours.

The selector is stateful and streaming: feed it each generated token's text as
it is produced; it answers "read here?" without ever looking ahead -- so it can
run inside the live agent loop, and nothing about it depends on actions or step
alignment (the ground rule).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CUES: tuple[str, ...] = (
    "hmm", "wait", "let me", "actually", "should i", "alternatively",
)

_WS = re.compile(r"\s+")
_TRAILING_NON_ALNUM = re.compile(r"[^a-z0-9]+$")


@dataclass(frozen=True)
class ReadTrigger:
    """One selected read position (0-based index into the generated tokens)."""

    token_idx: int
    trigger: str  # "cadence" | "cue"
    cue: str | None = None


class StreamReadSelector:
    """Streaming decision: is generated-token position `k` a read position?

    - cadence: fires at every `cadence`-th token (positions K-1, 2K-1, ...).
    - cue: fires when the normalized text generated so far -- lowercased,
      whitespace collapsed, trailing punctuation stripped -- ends with a cue at
      a word boundary. Matching is on detokenized text, so cues spanning token
      boundaries ("let" + " me") fire, and words merely containing a cue
      ("awaits", "hmmm") do not.
    - one read per position; if both fire, the read is recorded as "cue".
    """

    def __init__(self, cadence: int = 32, cues: tuple[str, ...] = DEFAULT_CUES,
                 max_buffer: int = 256):
        if cadence < 1:
            raise ValueError("cadence must be >= 1")
        self.cadence = cadence
        self.cues = tuple(_WS.sub(" ", c.lower()).strip() for c in cues)
        if any(not c for c in self.cues):
            raise ValueError("empty cue")
        self._max_buffer = max(max_buffer, 4 * max((len(c) for c in self.cues), default=8))
        self._buf = ""
        self._idx = -1

    def step(self, token_text: str) -> ReadTrigger | None:
        """Advance one generated token; return the read to take here, if any."""
        self._idx += 1
        self._buf = (self._buf + token_text.lower())[-self._max_buffer :]

        cue_hit: str | None = None
        # A cue's last character is always alphanumeric, so only a token that
        # adds word characters can complete one; punctuation-only and empty
        # tokens are skipped, which is also the refire guard ("wait" then ","
        # must not fire twice). Buffer LENGTH is not a valid proxy for "new
        # text": once the rolling buffer saturates it stays constant while
        # text keeps streaming.
        if any(c.isalnum() for c in token_text):
            base = _TRAILING_NON_ALNUM.sub("", _WS.sub(" ", self._buf))
            for cue in self.cues:
                if base.endswith(cue):
                    before = len(base) - len(cue) - 1
                    if before < 0 or not base[before].isalnum():
                        cue_hit = cue
                        break

        if cue_hit is not None:
            return ReadTrigger(self._idx, "cue", cue_hit)
        if (self._idx + 1) % self.cadence == 0:
            return ReadTrigger(self._idx, "cadence")
        return None


def read_positions(tokens: list[str], cadence: int = 32,
                   cues: tuple[str, ...] = DEFAULT_CUES) -> list[ReadTrigger]:
    """Batch helper over a full token list (offline replay / tests)."""
    sel = StreamReadSelector(cadence=cadence, cues=cues)
    out = []
    for tok in tokens:
        hit = sel.step(tok)
        if hit is not None:
            out.append(hit)
    return out
