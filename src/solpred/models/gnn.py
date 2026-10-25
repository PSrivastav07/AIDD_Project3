"""Directed message-passing neural network (D-MPNN; Yang et al. 2019, J. Chem. Inf. Model. 59, 3370)
in plain PyTorch - no torch_geometric / dgl / chemprop dependency.

Method
------
Hidden states live on *directed* bonds. With atom features x_v, bond features e_vw, and N(v) the
neighbours of v (``depth`` = T, counted as in Chemprop's implementation: T-1 edge updates followed
by the atom-level aggregation, so the receptive field is T bonds)::

    a_vw   = W_i [x_v, e_vw]                       (pre-activation input, kept for the skip)
    h0_vw  = ReLU(a_vw)
    m_vw   = sum_{k in N(v)} h_kv  -  h_wv          (reverse-edge subtraction: no echo back to w)
    h_vw   = ReLU(a_vw + W_m m_vw)                  (t = 1 .. T-1)
    m_v    = sum_{k in N(v)} h_kv
    h_v    = ReLU(W_o [x_v, m_v])                   (atoms without bonds, e.g. Na+, get m_v = 0
                                                     but still contribute through x_v)
    h_mol  = mean_v h_v  |  sum_v h_v  |  sum_v h_v / 100 ("norm")
    y      = FFN([h_mol, d_mol])                    (d_mol: optional standardised RDKit descriptors)

Implementation notes (CPU throughput)
    * Graphs come from ``solpred.features.graph.GraphStore`` (featurised once, cached on disk);
      a mini-batch is built from vectorised index arrays and gathered from dataset-wide feature
      tensors - no per-molecule Python work inside the training loop. The edge inputs
      ``[x_v, e_vw]`` are precomputed for the whole dataset, so no gradient flows through a gather.
    * Every neighbourhood sum is a sparse (CSR) matrix product: the message step is
      ``m = K h`` with the edge-to-edge operator K, which already excludes the reverse edge
      (``K[vw, kv] = 1`` for k != w); the atom aggregation is ``m_atom = B h`` and the readout
      ``h_mol = P h_atom`` (P holds 1/n_atoms for the mean). Each operator carries its transpose so
      the backward pass is another sparse product (``_SpMM``). On a contended CPU this was ~3-5x
      faster than ``index_add_`` scatter-adds, which serialise in deterministic mode.
    * Validation / test inference uses ``torch.inference_mode`` with large pre-collated batches.

Training (``train_dmpnn``): Adam, MSE on train-standardised targets, Noam-style learning-rate
schedule (linear warm-up then exponential decay, as in Chemprop), early stopping on the validation
MAE (in log units, dropout off) with the best-epoch weights restored. The test set is never seen.

Dropout and uncertainty: ``dropout`` acts before every linear layer of the FFN head;
``encoder_dropout`` (default 0, as in Chemprop) after the message-passing steps and atom update.
``mc_dropout_std`` implements Monte-Carlo dropout (Gal & Ghahramani 2016): the dropout layers are
kept active at inference and the std over ``mc_passes`` stochastic passes is returned. With a
deterministic encoder the molecule embedding is computed once and the passes are a single batched
FFN evaluation on ``mc_passes`` stacked copies. Point predictions are the deterministic
(dropout-off) outputs.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from solpred.features.graph import ATOM_FDIM, BOND_FDIM, GraphStore

warnings.filterwarnings("ignore", message=".*[Ss]parse CSR tensor support is in beta.*")


# --------------------------------------------------------------------------- configuration
@dataclass
class GNNConfig:
    """Hyper-parameters of the D-MPNN and its training loop."""
    hidden: int = 300
    depth: int = 3
    ffn_hidden: int = 300
    ffn_layers: int = 2
    dropout: float = 0.1               # FFN head (also the Monte-Carlo dropout rate)
    encoder_dropout: float = 0.0       # message passing + atom update
    aggregation: str = "mean"          # "mean" | "sum" | "norm" (sum / 100)
    batch_size: int = 64
    eval_batch_size: int = 1024
    max_epochs: int = 50
    patience: int = 10
    warmup_epochs: float = 2.0
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 10.0
    mc_passes: int = 10
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GNNConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- scalers
class TargetScaler:
    """Standardise targets with the training mean / std."""

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y)) or 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64) - self.mean_) / self.std_

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std_ + self.mean_

    def to_dict(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetScaler":
        s = cls()
        s.mean_, s.std_ = float(d["mean"]), float(d["std"])
        return s


class DescriptorScaler:
    """Median imputation + standardisation fitted on the training rows, then clipping to +-``clip``.

    Columns that are constant (or entirely missing) in the training set are mapped to 0.
    """

    def __init__(self, clip: float = 10.0):
        self.clip = clip

    def fit(self, X: np.ndarray) -> "DescriptorScaler":
        X = np.asarray(X, dtype=np.float64)
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.median_ = np.where(np.isfinite(med), med, 0.0)
        Xi = self._impute(X)
        self.mean_ = Xi.mean(0)
        sd = Xi.std(0)
        self.std_ = np.where(sd > 1e-12, sd, np.inf)
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        bad = ~np.isfinite(X)
        X[bad] = np.take(self.median_, np.where(bad)[1])
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (self._impute(X) - self.mean_) / self.std_
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)


# --------------------------------------------------------------------------- batches
@dataclass
class GraphSet:
    """A set of molecules (rows of a ``GraphStore``) with aligned targets and optional descriptors."""
    pos: np.ndarray
    y: np.ndarray | None = None
    desc: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.pos)


class SparseOp:
    """A fixed sparse matrix ``A`` (CSR) with its transpose, applied as ``A @ h`` with exact autograd."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple[int, int]):
        self.A = self._csr(rows, cols, vals, shape)
        self.AT = self._csr(cols, rows, vals, (shape[1], shape[0]))

    @staticmethod
    def _csr(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple[int, int]) -> torch.Tensor:
        o = np.argsort(rows, kind="stable")
        crow = np.zeros(shape[0] + 1, dtype=np.int32)
        np.cumsum(np.bincount(rows, minlength=shape[0]), out=crow[1:])
        return torch.sparse_csr_tensor(torch.from_numpy(crow), torch.from_numpy(cols[o].astype(np.int32)),
                                       torch.from_numpy(vals[o].astype(np.float32)), size=shape,
                                       check_invariants=False)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        return _SpMM.apply(self.A, self.AT, h)


class _SpMM(torch.autograd.Function):
    """``A @ h`` (MKL CSR kernel) whose gradient w.r.t. ``h`` is ``A^T @ grad`` - no scatter-add."""

    @staticmethod
    def forward(ctx, A: torch.Tensor, AT: torch.Tensor, h: torch.Tensor) -> torch.Tensor:  # noqa: N803
        ctx.AT = AT
        return torch.sparse.mm(A, h)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return None, None, torch.sparse.mm(ctx.AT, g.contiguous())


@dataclass
class Batch:
    """Torch tensors of one mini-batch (see ``GraphTensors.batch``)."""
    x: torch.Tensor           # [n_atoms, ATOM_FDIM]
    xe: torch.Tensor          # [n_edges, ATOM_FDIM + BOND_FDIM] = [x_src, e_vw] per directed edge
    msg: SparseOp             # [n_edges, n_edges] message operator K (reverse edge excluded)
    incoming: SparseOp        # [n_atoms, n_edges] sum over edges pointing into each atom
    pool: SparseOp            # [n_mols, n_atoms] readout weights (1/n_atoms for mean)
    n_mols: int
    desc: torch.Tensor | None = None

    @property
    def n_edges(self) -> int:
        return self.xe.shape[0]


class GraphTensors:
    """Dataset-wide feature tensors (atoms; directed-edge inputs [x_src, e]); batches gather from them."""

    def __init__(self, store: GraphStore, aggregation: str = "mean"):
        self.store, self.aggregation = store, aggregation
        self.x = torch.from_numpy(np.ascontiguousarray(store.x))
        self.xe = torch.from_numpy(np.ascontiguousarray(
            np.concatenate([store.x[store.edge_src_global()], store.e], axis=1)))

    def _pool_weights(self, n_atoms: np.ndarray) -> np.ndarray:
        if self.aggregation == "mean":
            return 1.0 / np.maximum(n_atoms, 1)
        return np.full(len(n_atoms), 0.01 if self.aggregation == "norm" else 1.0)

    def batch(self, mols: np.ndarray, desc: np.ndarray | None = None) -> Batch:
        """Collate the molecules at store positions ``mols`` (descriptor rows aligned with ``mols``)."""
        b = self.store.batch(mols)
        A, E = b.n_atoms, b.n_edges
        w = self._pool_weights(self.store.n_atoms_per_mol[np.asarray(mols)])
        return Batch(x=self.x.index_select(0, torch.from_numpy(b.atom_idx)),
                     xe=self.xe.index_select(0, torch.from_numpy(b.edge_idx)),
                     msg=SparseOp(b.msg_rows, b.msg_cols, np.ones(len(b.msg_rows)), (E, E)),
                     incoming=SparseOp(b.dst, np.arange(E), np.ones(E), (A, E)),
                     pool=SparseOp(b.atom2mol, np.arange(A), w[b.atom2mol], (b.n_mols, A)),
                     n_mols=b.n_mols, desc=None if desc is None else torch.from_numpy(desc))

    def iter_batches(self, gs: GraphSet, batch_size: int, order: np.ndarray | None = None) -> Iterator[tuple[np.ndarray, Batch]]:
        """Yield ``(row_indices_into_gs, Batch)``; ``order`` permutes the rows (training shuffles)."""
        rows = np.arange(len(gs)) if order is None else order
        for s in range(0, len(rows), batch_size):
            r = rows[s:s + batch_size]
            yield r, self.batch(gs.pos[r], None if gs.desc is None else gs.desc[r])

    def precollate(self, gs: GraphSet, batch_size: int) -> list[Batch]:
        """Fixed batches for repeated evaluation (validation every epoch)."""
        return [b for _, b in self.iter_batches(gs, batch_size)]


# --------------------------------------------------------------------------- model
class DMPNNEncoder(nn.Module):
    """Directed-bond message passing followed by atom-level aggregation and a molecular readout."""

    def __init__(self, hidden: int = 300, depth: int = 3, dropout: float = 0.0, aggregation: str = "mean",
                 atom_fdim: int = ATOM_FDIM, bond_fdim: int = BOND_FDIM):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if aggregation not in ("mean", "sum", "norm"):
            raise ValueError(f"unknown aggregation {aggregation!r}")
        self.hidden, self.depth, self.aggregation = hidden, depth, aggregation
        self.W_i = nn.Linear(atom_fdim + bond_fdim, hidden, bias=False)
        self.W_m = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(atom_fdim + hidden, hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, b: Batch) -> torch.Tensor:
        """Molecule embeddings [n_mols, hidden]; the readout weights (mean/sum/norm) live in ``b.pool``."""
        if b.n_edges == 0:  # only bond-less species in the batch (e.g. single ions)
            m_atom = b.x.new_zeros(b.x.shape[0], self.hidden)
        else:
            a = self.W_i(b.xe)                                  # a_vw = W_i [x_v, e_vw]
            h = self.drop(F.relu(a))
            for _ in range(self.depth - 1):                     # h_vw = ReLU(a_vw + W_m (K h)_vw)
                h = self.drop(F.relu(torch.addmm(a, b.msg(h), self.W_m.weight.t())))
            m_atom = b.incoming(h)                              # m_v = sum_k h_kv
        h_atom = self.drop(F.relu(self.W_o(torch.cat([b.x, m_atom], dim=1))))
        return b.pool(h_atom)


class DMPNN(nn.Module):
    """D-MPNN encoder + feed-forward head; ``n_desc`` > 0 appends molecule-level descriptors."""

    def __init__(self, cfg: GNNConfig, n_desc: int = 0):
        super().__init__()
        self.n_desc = n_desc
        self.encoder = DMPNNEncoder(cfg.hidden, cfg.depth, cfg.encoder_dropout, cfg.aggregation)
        layers: list[nn.Module] = []
        d_in = cfg.hidden + n_desc
        for _ in range(cfg.ffn_layers - 1):
            layers += [nn.Dropout(cfg.dropout), nn.Linear(d_in, cfg.ffn_hidden), nn.ReLU()]
            d_in = cfg.ffn_hidden
        layers += [nn.Dropout(cfg.dropout), nn.Linear(d_in, 1)]
        self.ffn = nn.Sequential(*layers)

    def readout(self, b: Batch) -> torch.Tensor:
        """Molecule vector fed to the FFN: encoder output (+ descriptors)."""
        z = self.encoder(b)
        return torch.cat([z, b.desc], dim=1) if self.n_desc else z

    def forward(self, b: Batch) -> torch.Tensor:
        return self.ffn(self.readout(b)).squeeze(1)


def n_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class NoamLR:
    """Linear warm-up from ``init_lr`` to ``max_lr``, then exponential decay to ``final_lr``."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, init_lr: float,
                 max_lr: float, final_lr: float):
        self.opt, self.warm, self.total = optimizer, max(1, warmup_steps), max(total_steps, warmup_steps + 1)
        self.init_lr, self.max_lr, self.final_lr = init_lr, max_lr, final_lr
        self.gamma = (final_lr / max_lr) ** (1.0 / (self.total - self.warm))
        self.step_num = 0
        self._set(self.lr())

    def lr(self) -> float:
        s = self.step_num
        if s < self.warm:
            return self.init_lr + (self.max_lr - self.init_lr) * s / self.warm
        return self.max_lr * self.gamma ** (s - self.warm)

    def _set(self, lr: float) -> None:
        for gr in self.opt.param_groups:
            gr["lr"] = lr

    def step(self) -> None:
        self.step_num += 1
        self._set(self.lr())


# --------------------------------------------------------------------------- inference
@torch.inference_mode()
def embed(model: DMPNN, batches: list[Batch]) -> torch.Tensor:
    """FFN input (readout + descriptors) for pre-collated batches, encoder in eval mode."""
    model.eval()
    return torch.cat([model.readout(b) for b in batches])


@torch.inference_mode()
def predict_from_embedding(model: DMPNN, z: torch.Tensor, scaler: TargetScaler) -> np.ndarray:
    """Deterministic (dropout-off) predictions in original units."""
    model.eval()
    return scaler.inverse(model.ffn(z).squeeze(1).numpy())


@torch.inference_mode()
def predict(model: DMPNN, data: GraphTensors, gs: GraphSet, scaler: TargetScaler,
            batch_size: int = 1024) -> np.ndarray:
    """Deterministic predictions for a ``GraphSet`` (original units)."""
    return predict_from_embedding(model, embed(model, data.precollate(gs, batch_size)), scaler)


@torch.inference_mode()
def mc_dropout_std(model: DMPNN, data: GraphTensors, gs: GraphSet, scaler: TargetScaler, n_passes: int = 10,
                   seed: int = 0, batch_size: int = 1024, z: torch.Tensor | None = None) -> np.ndarray:
    """Std (original units) over ``n_passes`` forward passes with dropout active.

    Deterministic encoder: the embedding ``z`` (computed here unless given) is stacked ``n_passes``
    times and pushed through the FFN in train mode in one call. Stochastic encoder: full passes.
    """
    torch.manual_seed(10_000 + seed)
    if isinstance(model.encoder.drop, nn.Dropout):
        batches = data.precollate(gs, batch_size)
        model.train()
        passes = torch.stack([torch.cat([model(b) for b in batches]) for _ in range(n_passes)])
    else:
        z = embed(model, data.precollate(gs, batch_size)) if z is None else z
        model.ffn.train()
        passes = model.ffn(z.repeat(n_passes, 1)).view(n_passes, -1)
    model.eval()
    return scaler.inverse(passes.numpy()).std(0)


# --------------------------------------------------------------------------- training
def set_determinism(seed: int, threads: int | None = None) -> None:
    """Seed torch / numpy, fix the intra-op thread count and request deterministic kernels."""
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _train_epoch(model: DMPNN, data: GraphTensors, train: GraphSet, z_train: torch.Tensor, cfg: GNNConfig,
                 opt: torch.optim.Optimizer, sched: NoamLR, rng: np.random.RandomState) -> float:
    """One pass over shuffled training batches; returns the mean training MSE (standardised units)."""
    model.train()
    total = torch.zeros(())
    for rows, b in data.iter_batches(train, cfg.batch_size, rng.permutation(len(train))):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(b), z_train[rows])
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        total += loss.detach() * len(rows)
    return float(total) / len(train)


def train_dmpnn(cfg: GNNConfig, data: GraphTensors, train: GraphSet, val: GraphSet, seed: int = 0,
                max_epochs: int | None = None, log: Callable[[str], None] = print
                ) -> tuple[DMPNN, TargetScaler, pd.DataFrame, dict]:
    """Train a D-MPNN with early stopping on validation MAE (log units).

    Returns ``(model_with_best_weights, target_scaler, per_epoch_history, info)``.
    """
    if data.aggregation != cfg.aggregation:
        raise ValueError(f"GraphTensors built for {data.aggregation!r} readout, config asks {cfg.aggregation!r}")
    set_determinism(seed)
    scaler = TargetScaler().fit(train.y)
    z_train = torch.as_tensor(scaler.transform(train.y), dtype=torch.float32)
    n_desc = 0 if train.desc is None else train.desc.shape[1]
    model = DMPNN(cfg, n_desc=n_desc)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.init_lr, weight_decay=cfg.weight_decay, fused=True)
    epochs = max_epochs or cfg.max_epochs
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    sched = NoamLR(opt, int(cfg.warmup_epochs * steps_per_epoch), cfg.max_epochs * steps_per_epoch,
                   cfg.init_lr, cfg.max_lr, cfg.final_lr)
    rng = np.random.RandomState(seed)
    val_batches = data.precollate(val, cfg.eval_batch_size)
    history: list[dict] = []
    best = {"val_MAE": np.inf, "epoch": -1, "state": None}
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_epoch(model, data, train, z_train, cfg, opt, sched, rng)
        t_train = time.time() - t0
        err = predict_from_embedding(model, embed(model, val_batches), scaler) - val.y
        row = {"epoch": epoch, "train_loss_mse_std": train_loss, "val_MAE": float(np.mean(np.abs(err))),
               "val_RMSE": float(np.sqrt(np.mean(err ** 2))), "lr_end": sched.lr(),
               "train_seconds": t_train, "epoch_seconds": time.time() - t0}
        history.append(row)
        if row["val_MAE"] < best["val_MAE"] - 1e-6:
            best = {"val_MAE": row["val_MAE"], "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        log(f"  epoch {epoch:3d}  train_mse(std)={train_loss:.4f}  val_MAE={row['val_MAE']:.4f}"
            f"  lr={row['lr_end']:.2e}  ({row['epoch_seconds']:.1f}s)")
        if epoch - best["epoch"] >= cfg.patience:
            break
    model.load_state_dict(best["state"])
    model.eval()
    info = {"best_epoch": best["epoch"], "best_val_MAE": best["val_MAE"], "epochs_run": len(history),
            "train_seconds": time.time() - t_start, "n_parameters": n_parameters(model), "n_desc": n_desc,
            "early_stopped": len(history) < epochs, "threads": torch.get_num_threads()}
    return model, scaler, pd.DataFrame(history), info


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(model: DMPNN, scaler: TargetScaler, cfg: GNNConfig, out_dir: Path, tag: str,
                    extra: dict | None = None) -> list[Path]:
    """``<tag>.pt`` (state_dict), ``<tag>_hparams.json`` and ``<tag>_target_scaler.json``."""
    from solpred.config import save_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [out_dir / f"{tag}.pt", out_dir / f"{tag}_hparams.json", out_dir / f"{tag}_target_scaler.json"]
    torch.save(model.state_dict(), files[0])
    save_json({"model_class": "solpred.models.gnn.DMPNN", "n_desc": model.n_desc, "config": cfg.to_dict(),
               "atom_fdim": ATOM_FDIM, "bond_fdim": BOND_FDIM, "featurizer": "solpred.features.graph",
               "torch_version": torch.__version__, **(extra or {})}, files[1])
    save_json({"type": "standardise", **scaler.to_dict(),
               "note": "logS = model_output * std + mean (training-set statistics)"}, files[2])
    return files


def load_checkpoint(out_dir: Path, tag: str) -> tuple[DMPNN, TargetScaler, GNNConfig]:
    """Inverse of ``save_checkpoint`` (model in eval mode)."""
    from solpred.config import load_json

    out_dir = Path(out_dir)
    hp = load_json(out_dir / f"{tag}_hparams.json")
    cfg = GNNConfig.from_dict(hp["config"])
    model = DMPNN(cfg, n_desc=int(hp["n_desc"]))
    model.load_state_dict(torch.load(out_dir / f"{tag}.pt", weights_only=True))
    model.eval()
    return model, TargetScaler.from_dict(load_json(out_dir / f"{tag}_target_scaler.json")), cfg
