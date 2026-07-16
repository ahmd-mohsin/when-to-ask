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

# Multi-digit literals (ints/floats), not bare single digits -- loop indices
# and list numbering would flood reads otherwise (decisions/016).
DEFAULT_VALUE_PATTERN = r"\d{2,}(?:\.\d+)?"

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
    - value reads (decisions/016, off by default): when `value_pattern` is set,
      a read also fires the moment the generated delta emits a matching literal
      (e.g. a number) -- the instant a value-fork's distinguishing token is
      written, which cadence reads mostly straddle. A cooldown bounds the rate
      (code is full of digits). Still reading DURING generation, never at the
      action boundary -- this is a cue-set extension per decisions/006.
    - one read per position; priority cue > value > cadence.
    """

    def __init__(self, cadence: int = 32, cues: tuple[str, ...] = DEFAULT_CUES,
                 max_buffer: int = 256, value_pattern: str | None = None,
                 value_cooldown: int = 8):
        if cadence < 1:
            raise ValueError("cadence must be >= 1")
        self.cadence = cadence
        self.cues = tuple(_WS.sub(" ", c.lower()).strip() for c in cues)
        if any(not c for c in self.cues):
            raise ValueError("empty cue")
        self._value_re = re.compile(value_pattern) if value_pattern else None
        self._value_cooldown = max(0, value_cooldown)
        self._last_value_read = -10**9
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
        if self._value_re is not None and token_text:
            # Match against the rolling buffer, not the lone token: tokenizers
            # that split literals across tokens (Qwen emits numbers digit by
            # digit) never put >= 2 digits in one token, so a token-local
            # search can never fire. A read fires when a match ENDS inside the
            # text this token added -- the instant the literal (first) becomes
            # matchable -- and the cooldown absorbs refires as the same
            # literal keeps extending.
            tail = self._buf[-64:]
            added_from = max(len(tail) - len(token_text), 0)
            m = None
            for m_ in self._value_re.finditer(tail):
                if m_.end() > added_from:
                    m = m_
            if m and (self._idx - self._last_value_read) >= self._value_cooldown:
                self._last_value_read = self._idx
                return ReadTrigger(self._idx, "value", m.group(0))
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
