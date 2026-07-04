"""When-to-Ask detector (method: `when-to-ask-offline-online (1).md`).

Phase-1 surface. `wta.hf_reader` is intentionally NOT imported here: it needs
torch and must stay off the CPU/laptop import path (decisions/004).
"""

from wta.a1_direction import ambiguity_signal, auroc, build_direction  # noqa: F401
from wta.logging_schema import ActionEvent, ReadRecord, RunLog  # noqa: F401
from wta.reads import DEFAULT_CUES, ReadTrigger, StreamReadSelector  # noqa: F401
