"""A2: the disentangling autoencoder -- topic ``T(h)`` and resolution lean ``L(h)``.

Architecture follows **ReDAct** (arXiv 2602.19396) -- no public code exists
(verified; decisions/002), so the shared-body/two-head encoder + decoder and
the gradient-reversal invariance pull are implemented from the paper and cited
as "following ReDAct".

MIGRATED from **REAL** (third_party/REAL_ICLR @ c93c04f, MIT;
``validation/OneForAll.py``): ``build_mlp``, ``supervised_contrastive_loss``
(the InfoNCE-style supervised term), ``set_global_determinism`` (CPU-adapted).

OURS (build brief migrate-vs-build table): the supervision targets -- decision
identity for ``T``, interpretation *class* for ``L`` (categorical, never a
signed scalar) -- and the batch-decorrelation form of the orthogonality pull.

This module needs torch (CPU is fine; decisions/011). It is imported by
training / gates / smoke paths only -- never by ``wta/__init__``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Migrated pieces (REAL, validation/OneForAll.py)
# ---------------------------------------------------------------------------


def set_global_determinism(seed: int) -> None:
    """REAL's determinism helper, CPU-adapted (CUDA/TF32 knobs dropped)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def build_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int,
              slope: float = 1e-2) -> nn.Sequential:
    """REAL's MLP builder (Linear + LeakyReLU stack), verbatim port."""
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h), nn.LeakyReLU(negative_slope=slope, inplace=True)]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.07) -> torch.Tensor:
    """REAL's supervised contrastive (InfoNCE-style) loss, verbatim port."""
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / temperature
    labels = labels.view(-1, 1)
    mask = (labels == labels.t()).float()
    self_mask = torch.eye(z.size(0), device=z.device)
    mask = mask * (1 - self_mask)
    log_prob = sim - sim.logsumexp(dim=1, keepdim=True)
    denom = mask.sum(dim=1).clamp_min(1.0)
    pos_log_prob = (mask * log_prob).sum(dim=1) / denom
    return -pos_log_prob.mean()


# ---------------------------------------------------------------------------
# Gradient reversal (standard construction; the invariance pull is ReDAct's)
# ---------------------------------------------------------------------------


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class A2Config:
    in_dim: int
    n_topics: int
    n_classes: int
    d_topic: int = 16
    d_lean: int = 8
    hidden: int = 128
    epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 0
    w_topic: float = 1.0
    w_supcon: float = 0.5
    w_lean: float = 1.0
    w_adv: float = 2.0
    grl_max: float = 1.0
    # Adversary catch-up steps per encoder step. With a single joint optimizer
    # the adversary lags and the encoder learns to fool its CURRENT weights
    # while the class stays linearly readable from T (measured: defeated-MLP
    # adversary at chance, fresh linear probe at 1.00). Keeping the adversary
    # near-optimal makes the reversed gradient point at information that is
    # actually present.
    adv_steps: int = 3
    # OURS (addition to ReDAct's recipe, flagged in spec A2): direct
    # conditional-mean alignment. Measured on fixtures, GRL alone loses the
    # arms race -- the encoder perturbs T against the adversary's current
    # weights while the class subspace stays linearly readable (probe 1.00 at
    # w_adv up to 8). Collapsing within-topic class-conditional means of T
    # removes exactly what a linear probe reads, and IS the fork-collocation
    # property expressed as a loss; GRL remains for nonlinear residue.
    w_condmean: float = 4.0
    w_ortho: float = 1.0
    w_recon: float = 1.0


class DisentangleAE(nn.Module):
    """Shared 2-layer body -> T/L heads; decoder rebuilds h from (T, L).

    Training-only auxiliaries: linear topic/lean classifiers and the adversary
    that reads the lean class from T through gradient reversal.
    """

    def __init__(self, cfg: A2Config):
        super().__init__()
        self.cfg = cfg
        self.body = build_mlp(cfg.in_dim, (cfg.hidden,), cfg.hidden)
        self.head_topic = nn.Linear(cfg.hidden, cfg.d_topic)
        self.head_lean = nn.Linear(cfg.hidden, cfg.d_lean)
        self.decoder = build_mlp(cfg.d_topic + cfg.d_lean, (cfg.hidden,), cfg.in_dim)
        self.topic_cls = nn.Linear(cfg.d_topic, cfg.n_topics)
        self.lean_cls = nn.Linear(cfg.d_lean, cfg.n_classes)
        self.adversary = build_mlp(cfg.d_topic, (cfg.hidden // 2,), cfg.n_classes)

    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.body(h)
        return self.head_topic(z), self.head_lean(z)

    def forward(self, h: torch.Tensor):
        t, l = self.encode(h)
        h_hat = self.decoder(torch.cat([t, l], dim=-1))
        return t, l, h_hat


def compute_losses(model: DisentangleAE, h: torch.Tensor, topic_y: torch.Tensor,
                   class_y: torch.Tensor, grl_lam: float) -> dict[str, torch.Tensor]:
    """The four pulls (spec A2). ``class_y == -1`` marks reads with no
    interpretation-class label (clear decisions); pulls 2 and 3 skip them."""
    cfg = model.cfg
    t, l, h_hat = model(h)
    labeled = class_y >= 0

    losses = {
        "topic": F.cross_entropy(model.topic_cls(t), topic_y),
        "supcon": supervised_contrastive_loss(t, topic_y),
        "recon": F.mse_loss(h_hat, h),
    }
    if labeled.any():
        losses["lean"] = F.cross_entropy(model.lean_cls(l[labeled]), class_y[labeled])
        losses["adv"] = F.cross_entropy(
            model.adversary(grad_reverse(t[labeled], grl_lam)), class_y[labeled]
        )
        # Conditional-mean alignment (OURS; see A2Config.w_condmean): within
        # each decision, the class-conditional means of T collapse onto the
        # decision mean -- forking runs collocate by construction.
        t_lab, ty_lab, cy_lab = t[labeled], topic_y[labeled], class_y[labeled]
        align, groups = torch.zeros((), device=h.device), 0
        for tt in torch.unique(ty_lab):
            m_t = ty_lab == tt
            mu_t = t_lab[m_t].mean(dim=0)
            for cc in torch.unique(cy_lab[m_t]):
                m_tc = m_t & (cy_lab == cc)
                if int(m_tc.sum()) >= 2:
                    align = align + ((t_lab[m_tc].mean(dim=0) - mu_t) ** 2).sum()
                    groups += 1
        losses["condmean"] = align / max(groups, 1)
    else:
        zero = torch.zeros((), device=h.device)
        losses["lean"], losses["adv"], losses["condmean"] = zero, zero, zero

    # Orthogonality as batch decorrelation between T and L features (T and L
    # have different dims, so per-sample cosine is undefined; ReDAct's "keep
    # them pointing in different directions" becomes cross-correlation ~ 0).
    t_c = t - t.mean(dim=0, keepdim=True)
    l_c = l - l.mean(dim=0, keepdim=True)
    corr = (t_c.t() @ l_c) / max(1, h.shape[0])
    losses["ortho"] = (corr ** 2).mean()

    losses["total"] = (
        cfg.w_topic * losses["topic"] + cfg.w_supcon * losses["supcon"]
        + cfg.w_lean * losses["lean"] + cfg.w_adv * losses["adv"]
        + cfg.w_condmean * losses["condmean"]
        + cfg.w_ortho * losses["ortho"] + cfg.w_recon * losses["recon"]
    )
    return losses


# ---------------------------------------------------------------------------
# Training + frozen numpy-facing wrapper
# ---------------------------------------------------------------------------


class A2Model:
    """Frozen encoders with a numpy API -- what offline stages hand to A3/A4/B."""

    def __init__(self, net: DisentangleAE):
        self.net = net.eval()
        self.cfg = net.cfg

    def _run(self, h: np.ndarray, fn) -> np.ndarray:
        arr = np.asarray(h, dtype=np.float32)
        flat = arr.reshape(-1, arr.shape[-1])
        with torch.no_grad():
            out = fn(torch.from_numpy(flat)).numpy()
        return out.reshape(*arr.shape[:-1], out.shape[-1]).astype(np.float32)

    def encode_topic(self, h: np.ndarray) -> np.ndarray:
        return self._run(h, lambda x: self.net.encode(x)[0])

    def encode_lean(self, h: np.ndarray) -> np.ndarray:
        return self._run(h, lambda x: self.net.encode(x)[1])

    def reconstruct(self, h: np.ndarray) -> np.ndarray:
        return self._run(h, lambda x: self.net(x)[2])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "A2Model":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        net = DisentangleAE(A2Config(**blob["cfg"]))
        net.load_state_dict(blob["state"])
        return cls(net)


def grl_diagnostic(model: A2Model, h: np.ndarray, class_labels: np.ndarray) -> dict:
    """The treadmill check: compare the TRAINED adversary's accuracy on T with
    a FRESH linear probe's. If the adversary is near chance but a fresh probe
    reads the class, the encoder only fooled the adversary's current weights
    (the failure measured on fixtures 2026-07-03) -- invariance did NOT happen.
    Run after every real training; report both numbers."""
    from sklearn.linear_model import LogisticRegression

    m = np.asarray(class_labels) >= 0
    if m.sum() < 10 or len(np.unique(class_labels[m])) < 2:
        return {"note": "too few class-labeled reads for the diagnostic"}
    t = model.encode_topic(np.asarray(h, dtype=np.float32)[m])
    y = np.asarray(class_labels)[m]
    with torch.no_grad():
        adv_acc = float((model.net.adversary(torch.from_numpy(t)).argmax(1).numpy() == y).mean())
    cut = int(0.7 * len(t))
    probe_acc = float(LogisticRegression(max_iter=1000)
                      .fit(t[:cut], y[:cut]).score(t[cut:], y[cut:])) if cut >= 10 else float("nan")
    return {"adversary_acc_on_T": adv_acc, "fresh_probe_acc_on_T": probe_acc,
            "chance": 1.0 / len(np.unique(y)),
            "treadmill_suspected": bool(probe_acc > adv_acc + 0.2)}


def train_a2(h: np.ndarray, topic_labels: np.ndarray, class_labels: np.ndarray,
             cfg: A2Config, log_every: int = 0,
             history_path: str | Path | None = None) -> A2Model:
    """Train on offline reads. ``class_labels`` uses -1 for unlabeled reads.

    Returns the frozen model. Deterministic for a given cfg.seed. With
    ``history_path``, per-epoch full-batch loss components are appended as
    JSONL -- the first thing to look at when a gate number looks wrong.
    """
    h = np.asarray(h, dtype=np.float32)
    if h.ndim != 2 or h.shape[1] != cfg.in_dim:
        raise ValueError(f"expected h (n, {cfg.in_dim}), got {h.shape}")
    if not (len(h) == len(topic_labels) == len(class_labels)):
        raise ValueError("h / topic_labels / class_labels length mismatch")

    set_global_determinism(cfg.seed)
    net = DisentangleAE(cfg)
    adv_params = list(net.adversary.parameters())
    adv_ids = {id(p) for p in adv_params}
    main_opt = torch.optim.Adam(
        [p for p in net.parameters() if id(p) not in adv_ids], lr=cfg.lr)
    adv_opt = torch.optim.Adam(adv_params, lr=cfg.lr)

    ht = torch.from_numpy(h)
    ty = torch.from_numpy(np.asarray(topic_labels, dtype=np.int64))
    cy = torch.from_numpy(np.asarray(class_labels, dtype=np.int64))
    n = len(ht)
    gen = torch.Generator().manual_seed(cfg.seed)

    net.train()
    for epoch in range(cfg.epochs):
        grl_lam = cfg.grl_max * (epoch + 1) / cfg.epochs  # 0 -> grl_max ramp
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            labeled = cy[idx] >= 0
            # (a) adversary catch-up: keep it near-optimal on the CURRENT T
            if labeled.any():
                for _ in range(cfg.adv_steps):
                    with torch.no_grad():
                        t_det = net.encode(ht[idx][labeled])[0]
                    adv_loss = F.cross_entropy(net.adversary(t_det), cy[idx][labeled])
                    adv_opt.zero_grad()
                    adv_loss.backward()
                    adv_opt.step()
            # (b) encoder/decoder step against the caught-up adversary (GRL)
            losses = compute_losses(net, ht[idx], ty[idx], cy[idx], grl_lam)
            main_opt.zero_grad()
            losses["total"].backward()
            main_opt.step()
        want_log = log_every and (epoch + 1) % log_every == 0
        if want_log or history_path:
            with torch.no_grad():
                full = compute_losses(net, ht, ty, cy, grl_lam)
            row = {k: round(float(v), 5) for k, v in full.items()}
            row["epoch"], row["grl_lam"] = epoch + 1, round(grl_lam, 3)
            if want_log:
                print("epoch", epoch + 1, json.dumps(row))
            if history_path:
                with Path(history_path).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

    return A2Model(net)
