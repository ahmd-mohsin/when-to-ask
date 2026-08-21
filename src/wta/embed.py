"""Local sentence embeddings for B2 v2 resolution vectors (027 Amendment A).

CPU-only, deterministic, no API: sentence-transformers/all-MiniLM-L6-v2 via
plain transformers (mean pooling over the attention mask, l2-normalized).
Loaded lazily so nothing in the contract suite pays the import cost.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"


class MiniLMEmbedder:
    def __init__(self, model_id: str = MODEL_ID, batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i:i + self.batch_size]
            enc = self.tok(chunk, padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
            with self._torch.no_grad():
                out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs = pooled.numpy().astype(np.float64)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-9, None)
            for t, v in zip(chunk, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])


class BgeEmbedder:
    """R5 (decisions/028 T1): BAAI/bge-large-en-v1.5 via plain transformers,
    CPU, deterministic. CLS-token pooling + l2 norm (the model card's
    documented sentence-embedding recipe; no query instruction — the pair
    task is symmetric). Truncation at 256 tokens mirrors MiniLMEmbedder so
    the encoder is the ONLY change between the R4 and R5 rows."""

    def __init__(self, model_id: str = BGE_MODEL_ID, batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i:i + self.batch_size]
            enc = self.tok(chunk, padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
            with self._torch.no_grad():
                out = self.model(**enc).last_hidden_state
            pooled = out[:, 0]
            vecs = pooled.numpy().astype(np.float64)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-9, None)
            for t, v in zip(chunk, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])
