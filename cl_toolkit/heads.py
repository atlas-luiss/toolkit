"""Classifier heads from the CL literature, usable with any strategy.

Two kinds:

* **Statistical heads** replace inference. After each task they absorb the
  task's (un-augmented) training features; at test time they score classes
  from those statistics. No gradient training, no forgetting of the head.

  ========  ===================================================================
  NCM       nearest class mean, cosine (iCaRL's NME; SimpleCIL, Zhou et al. 2024)
  FeCAM     per-class Mahalanobis with shrinkage + correlation normalisation
            (Goswami et al., NeurIPS 2023)
  LDA       shared-covariance Gaussian / streaming LDA (SLDA, Hayes & Kanan, CVPRW 2020)
  KLDA      LDA on random Fourier features of an RBF kernel
            (Momeni et al., arXiv 2024; the same recipe as RanDumb, Prabhu et al. 2024)
  RanPAC    random ReLU projection + ridge regression (McDonnell et al., NeurIPS 2023)
  ========  ===================================================================

* **Head corrections** keep the trained linear head and fix it after each task.

  ========  ===================================================================
  SLCA      classifier alignment: retrain the head on features sampled from stored
            class Gaussians (Zhang et al., ICCV 2023)
  WA        weight aligning: rescale new-class weight norms to the old ones
            (Zhao et al., CVPR 2020)
  ========  ===================================================================

The trainable cosine layer of LUCIR is :class:`cl_toolkit.models.base.CosineLinear`
(``classifier="cosine"``).

Use: ``WithHead(strategy, FeCAM())`` or ``clt.run(..., classifier="fecam")``.

All statistics are kept per *global* class id, so heads work in Class-, Domain-
and Task-IL (where scores are restricted to the task's classes).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

KINDS = ("class", "task", "domain")


class Head:
    """Base class. Override ``update`` + ``logits`` (statistical) or ``after_task`` (correction)."""

    scenarios = KINDS
    replaces_inference = True     # False: the model's trained head is used at test time
    needs_features = True         # False: ``update`` is not called (e.g. WA)

    def bind(self, model, scenario) -> None:
        self.model, self.scenario = model, scenario
        self.num_classes = scenario.benchmark.total_classes

    def update(self, feats: torch.Tensor, labels: torch.Tensor, task_id: int) -> None:
        """Absorb (N, D) float features with global labels of one task."""

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B, num_classes); -inf for classes never seen."""
        raise NotImplementedError

    def after_task(self, task_id: int, trainer) -> None:
        """Called after ``update`` at the end of each task."""

    @property
    def name(self):
        return type(self).__name__

    def __repr__(self):
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items()
                         if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))
                         and k not in ("num_classes",))
        return f"{self.name}({args})"


def _unseen_to_ninf(scores: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill((counts == 0).unsqueeze(0), float("-inf"))


# ---------------------------------------------------------------------------- NCM
class NCM(Head):
    """Nearest class mean. ``metric``: "cosine" (default) or "euclidean"."""

    def __init__(self, metric: str = "cosine"):
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.metric = metric
        self._sum = self._n = None

    def update(self, feats, labels, task_id):
        if self._sum is None:
            self._sum = feats.new_zeros(self.num_classes, feats.size(1))
            self._n = feats.new_zeros(self.num_classes)
        self._sum.index_add_(0, labels, feats)
        self._n.index_add_(0, labels, torch.ones_like(labels, dtype=feats.dtype))

    @property
    def means(self):
        return self._sum / self._n.clamp(min=1).unsqueeze(1)

    def logits(self, feats):
        mu = self.means
        if self.metric == "cosine":
            s = F.normalize(feats, dim=1) @ F.normalize(mu, dim=1).T
        else:
            s = -torch.cdist(feats, mu)
        return _unseen_to_ninf(s, self._n)


# ---------------------------------------------------------------------------- FeCAM
class FeCAM(Head):
    """Per-class Mahalanobis distance (Goswami et al., NeurIPS 2023).

    Features are optionally Tukey-transformed (x^beta; only valid for non-negative
    features, so ``tukey=None`` applies it automatically iff all features are >= 0,
    e.g. ReLU/ResNet features but not ViT [CLS]) and L2-normalised. Each class
    covariance is shrunk, ``S + g1*mean(diag)*I + g2*mean(offdiag)*(1-I)``, and
    normalised to a correlation matrix.

    Defaults are the authors' pre-trained-ViT script (FeCAM_vit_cifar100.py): shrink1=1,
    shrink2=0, no Tukey, L2-normalised features. Their ResNet experiments use shrink2=1
    and Tukey beta=0.5.
    """

    def __init__(self, shrink1: float = 1.0, shrink2: float = 0.0, tukey: Optional[bool] = None,
                 beta: float = 0.5, normalize: bool = True):
        self.shrink1, self.shrink2, self.tukey, self.beta, self.normalize = shrink1, shrink2, tukey, beta, normalize
        self._s1 = self._s2 = self._n = None
        self._mu = self._prec = None
        self._dirty = True

    def _transform(self, x):
        if self.tukey is None:
            self.tukey = bool((x >= 0).all())
        if self.tukey:
            x = x.clamp(min=0).pow(self.beta)
        return F.normalize(x, dim=1) if self.normalize else x

    def update(self, feats, labels, task_id):
        x = self._transform(feats).double()
        C, D = self.num_classes, x.size(1)
        if self._s1 is None:
            self._s1 = x.new_zeros(C, D)
            self._s2 = {}
            self._n = x.new_zeros(C)
        for c in labels.unique().tolist():
            xc = x[labels == c]
            self._s1[c] += xc.sum(0)
            self._s2[c] = self._s2.get(c, 0) + xc.T @ xc
            self._n[c] += len(xc)
        self._dirty = True

    def _finalize(self):
        seen = torch.nonzero(self._n).flatten().tolist()
        D = self._s1.size(1)
        eye = torch.eye(D, dtype=torch.float64, device=self._s1.device)
        mus, precs = [], []
        for c in seen:
            n = self._n[c]
            mu = self._s1[c] / n
            cov = self._s2[c] / n - torch.outer(mu, mu)
            diag_mean = cov.diagonal().mean()
            off_mean = (cov.sum() - cov.diagonal().sum()) / (D * D - D)
            cov = cov + self.shrink1 * diag_mean * eye + self.shrink2 * off_mean * (1 - eye)
            sd = cov.diagonal().clamp(min=1e-12).sqrt()
            cov = cov / torch.outer(sd, sd)                  # correlation normalisation
            mus.append(mu)
            precs.append(torch.linalg.pinv(cov, hermitian=True))
        self._seen = torch.tensor(seen, device=self._s1.device)
        self._mu = torch.stack(mus).float()
        self._prec = torch.stack(precs).float()
        self._dirty = False

    def logits(self, feats):
        if self._dirty:
            self._finalize()
        x = self._transform(feats.float())
        out = x.new_full((len(x), self.num_classes), float("-inf"))
        for i, c in enumerate(self._seen.tolist()):
            d = x - self._mu[i]
            out[:, c] = -((d @ self._prec[i]) * d).sum(1)
        return out


# ---------------------------------------------------------------------------- LDA / KLDA
class LDA(Head):
    """Gaussian classifier with one shared covariance (streaming LDA, SLDA).

    Exact under streaming: keeps per-class sums/counts and the global second
    moment, so classes may reappear across tasks (Domain-IL).
    ``shrinkage``: ridge added to the covariance, relative to its mean variance.
    """

    def __init__(self, shrinkage: float = 1e-2):
        self.shrinkage = shrinkage
        self._s1 = self._n = self._m2 = None
        self._W = self._b = None
        self._dirty = True

    def features(self, x):
        return x

    def update(self, feats, labels, task_id):
        z = self.features(feats.float()).double()
        if self._s1 is None:
            self._s1 = z.new_zeros(self.num_classes, z.size(1))
            self._n = z.new_zeros(self.num_classes)
            self._m2 = z.new_zeros(z.size(1), z.size(1))
        self._s1.index_add_(0, labels, z)
        self._n.index_add_(0, labels, torch.ones_like(labels, dtype=z.dtype))
        self._m2 += z.T @ z
        self._dirty = True

    def _finalize(self):
        n = self._n.clamp(min=1)
        mu = self._s1 / n.unsqueeze(1)
        N = self._n.sum()
        within = self._m2 - (mu * self._n.unsqueeze(1)).T @ mu          # sum_c n_c mu_c mu_c^T
        cov = within / N
        D = cov.size(0)
        cov = cov + self.shrinkage * cov.diagonal().mean() * torch.eye(D, dtype=cov.dtype, device=cov.device)
        P = torch.linalg.inv(cov)
        W = mu @ P                                                       # (C, D)
        self._W = W.float()
        self._b = (-0.5 * (W * mu).sum(1)).float()
        self._dirty = False

    def logits(self, feats):
        if self._dirty:
            self._finalize()
        s = self.features(feats.float()) @ self._W.T + self._b
        return _unseen_to_ninf(s, self._n)


class KLDA(LDA):
    """Kernel LDA via random Fourier features of an RBF kernel (Momeni et al., arXiv
    2412.15571, 2024; the same recipe as RanDumb, Prabhu et al., NeurIPS 2024).

    As in the paper / authors' code: RFF z(x) = sqrt(2/D) cos(w^T x + b), D = 5000, class
    means + one shared covariance, score = Gaussian log-likelihood (argmax-equivalent to
    their negative Mahalanobis distance). ``ensembles=5`` gives KLDA-E (average of the
    members' softmax probabilities, members seeded ``seed + i``).
    Differences: the paper tunes the kernel width per foundation model (no rule given);
    here ``gamma=None`` uses the median heuristic on the first task's training features.
    We add a small covariance shrinkage (the authors use a pseudo-inverse and none).
    """

    def __init__(self, rff_dim: int = 5000, gamma: Optional[float] = None, shrinkage: float = 1e-2, seed: int = 0,
                 ensembles: int = 1):
        super().__init__(shrinkage)
        self.rff_dim, self.gamma, self.seed, self.ensembles = rff_dim, gamma, seed, ensembles
        self._omega = self._phase = None
        self._members = [] if ensembles == 1 else \
            [KLDA(rff_dim, gamma, shrinkage, seed + i, ensembles=1) for i in range(ensembles)]

    def bind(self, model, scenario):
        super().bind(model, scenario)
        for m in self._members:
            m.bind(model, scenario)

    def update(self, feats, labels, task_id):
        if self._members:
            for m in self._members:
                m.update(feats, labels, task_id)
            self._n = self._members[0]._n
            return
        super().update(feats, labels, task_id)

    def logits(self, feats):
        if self._members:                                     # KLDA-E: mean probability
            p = torch.stack([torch.softmax(m.logits(feats), 1) for m in self._members]).mean(0)
            return _unseen_to_ninf(p.clamp(min=1e-30).log(), self._n)
        return super().logits(feats)

    def features(self, x):
        if self._omega is None:
            if self.gamma is None:
                sub = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(self.seed))[:1000].to(x.device)]
                self.gamma = float(1.0 / torch.pdist(sub).pow(2).median().clamp(min=1e-12))
            g = torch.Generator().manual_seed(self.seed)
            self._omega = (torch.randn(x.size(1), self.rff_dim, generator=g) * math.sqrt(2 * self.gamma)).to(x.device)
            self._phase = (torch.rand(self.rff_dim, generator=g) * 2 * math.pi).to(x.device)
        return math.sqrt(2.0 / self.rff_dim) * torch.cos(x @ self._omega + self._phase)


# ---------------------------------------------------------------------------- RanPAC
class RanPAC(Head):
    """Random projection (ReLU) + ridge regression on the accumulated Gram matrix.

    ``proj_dim``: random features M (paper: 10000). The ridge ``lambda`` is chosen after
    each task as in the paper (Appendix C.1) and the authors' code: fit on 80% of *that
    task's training features*, pick the lambda in {1e-8, ..., 1e8} with the lowest mean
    squared error on the other 20%. (Our split is a seeded random permutation; the
    authors take the first 80% of a shuffled loader.)
    """

    def __init__(self, proj_dim: int = 10000, lambdas: Optional[Sequence[float]] = None, seed: int = 0):
        self.proj_dim, self.seed = proj_dim, seed
        self.lambdas = list(lambdas) if lambdas is not None else [10.0 ** e for e in range(-8, 9)]
        self.ridge = None
        self._W = self._G = self._Q = self._Wout = None
        self._n = None

    def _h(self, f):
        return F.relu(f @ self._W)

    def update(self, feats, labels, task_id):
        C = self.num_classes
        if self._W is None:
            g = torch.Generator().manual_seed(self.seed)
            self._W = torch.randn(feats.size(1), self.proj_dim, generator=g).to(feats.device)
            self._G = feats.new_zeros(self.proj_dim, self.proj_dim, dtype=torch.float64)
            self._Q = feats.new_zeros(self.proj_dim, C, dtype=torch.float64)
            self._n = feats.new_zeros(C)
        H = self._h(feats.float()).double()
        Y = F.one_hot(labels, C).double()
        self.ridge = self._select_ridge(H, Y, labels)
        self._G += H.T @ H
        self._Q += H.T @ Y
        self._n.index_add_(0, labels, torch.ones_like(labels, dtype=feats.dtype))
        eye = torch.eye(self.proj_dim, device=H.device, dtype=torch.float64)
        self._Wout = torch.linalg.solve(self._G + self.ridge * eye, self._Q).float()

    def _select_ridge(self, H, Y, y):
        n = len(H)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed)).to(H.device)
        tr, va = perm[: int(0.8 * n)], perm[int(0.8 * n):]
        evals, evecs = torch.linalg.eigh(H[tr].T @ H[tr])
        VtQ = evecs.T @ (H[tr].T @ Y[tr])
        Hv = H[va] @ evecs
        best, best_lam = float("inf"), self.lambdas[0]
        for lam in self.lambdas:
            mse = (Hv @ (VtQ / (evals + lam).unsqueeze(1)) - Y[va]).pow(2).mean().item()
            if mse < best:
                best, best_lam = mse, lam
        return best_lam

    def logits(self, feats):
        return _unseen_to_ninf(self._h(feats.float()) @ self._Wout, self._n)


# ---------------------------------------------------------------------------- SLCA
class SLCA(Head):
    """Classifier alignment (Zhang et al., ICCV 2023), as in the authors' ``models/slca.py``.

    Stores a Gaussian (mean, full covariance + 1e-4 I) per class from its task's features.
    After every task t > 0 the model's shared head is retrained on features sampled from all
    seen classes' Gaussians: 256 fresh samples per class every epoch, 5 epochs, batches of
    256, SGD lr 0.01 (momentum 0.9, weight decay 5e-4, cosine). Class means of task k are
    scaled by ``0.9 + 0.1 (k+1)/(t+1)`` before sampling, and logits are normalised by the
    mean over tasks of each task's logit-slice L2 norm, with temperature 0.1.
    This is SLCA's classifier-alignment stage only; the "slow learner" stage (fine-tuning
    the backbone with a 100x smaller learning rate) is a separate choice of the strategy
    (e.g. ``freeze_backbone=False`` with a small lr).
    """

    scenarios = ("class", "domain")
    replaces_inference = False

    def __init__(self, samples_per_class: int = 256, epochs: int = 5, lr: float = 0.01, batch_size: int = 256,
                 logit_norm: Optional[float] = 0.1, shrinkage: float = 1e-4, weight_decay: float = 5e-4,
                 mean_decay: bool = True, seed: int = 0):
        self.samples_per_class, self.epochs, self.lr, self.batch_size = samples_per_class, epochs, lr, batch_size
        self.weight_decay, self.mean_decay = weight_decay, mean_decay
        self.logit_norm, self.shrinkage, self.seed = logit_norm, shrinkage, seed
        self._stats: Dict[int, tuple] = {}
        self._task_of: Dict[int, int] = {}

    def update(self, feats, labels, task_id):
        x = feats.float()
        D = x.size(1)
        for c in labels.unique().tolist():
            xc = x[labels == c].double()
            mu = xc.mean(0)
            cov = torch.cov(xc.T) if len(xc) > 1 else torch.zeros(D, D, dtype=xc.dtype, device=xc.device)
            cov = cov + self.shrinkage * torch.eye(D, dtype=cov.dtype, device=cov.device)
            self._stats[c] = (mu, torch.linalg.cholesky(cov))
            self._task_of.setdefault(c, task_id)

    def _sample(self, g, task_id):
        xs, ys = [], []
        for c, (mu, L) in self._stats.items():
            scale = 0.9 + 0.1 * (self._task_of[c] + 1) / (task_id + 1) if self.mean_decay else 1.0
            z = torch.randn(self.samples_per_class, len(mu), generator=g, device=mu.device, dtype=mu.dtype)
            xs.append((mu * scale + z @ L.T).float())
            ys.append(torch.full((self.samples_per_class,), c, device=mu.device))
        return torch.cat(xs), torch.cat(ys)

    def after_task(self, task_id, trainer):
        if task_id == 0:                                     # authors: alignment only from task 1 on
            return
        model = self.model
        head = model.get_head()
        g = torch.Generator(device=model._device).manual_seed(self.seed + task_id)
        opt = torch.optim.SGD(head.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        seen = self.scenario.seen_classes(task_id)
        dev = next(iter(self._stats.values()))[0].device
        col = torch.full((self.num_classes,), -1, dtype=torch.long, device=dev)
        col[torch.tensor(seen, device=dev)] = torch.arange(len(seen), device=dev)
        seen_t = torch.tensor(seen, device=dev)
        # columns (among seen) of each task, for the per-task logit norm
        slices = [col[torch.tensor(self.scenario.get_task(k).classes, device=dev)] for k in range(task_id + 1)]
        head.train()
        with torch.enable_grad(), model.class_mask(None):
            for _ in range(self.epochs):
                x, y = self._sample(g, task_id)              # fresh samples every epoch
                y = col[y]                                   # labels -> index among seen classes
                perm = torch.randperm(len(x), device=x.device)
                for i in range(0, len(x), self.batch_size):
                    b = perm[i:i + self.batch_size]
                    with trainer.autocast():                    # per step: no stale weight casts
                        logits = model.classify(x[b])[:, seen_t].float()   # seen columns only: no -inf
                    if self.logit_norm:
                        norms = torch.stack([logits[:, s].norm(dim=1) for s in slices], 1).mean(1, keepdim=True)
                        logits = logits / (norms + 1e-7) / self.logit_norm
                    loss = F.cross_entropy(logits, y[b])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                sched.step()
        head.eval()


# ---------------------------------------------------------------------------- WA
class WA(Head):
    """Weight aligning (Zhao et al., CVPR 2020): after each task t>0, scale the new
    classes' weight vectors so their mean norm matches the old classes'."""

    scenarios = ("class",)
    replaces_inference = False
    needs_features = False

    def after_task(self, task_id, trainer):
        if task_id == 0:
            return
        fc = self.model.get_head()._fc
        new = self.scenario.get_task(task_id).classes
        old = self.scenario.seen_classes(task_id - 1)
        with torch.no_grad():
            norms = fc.weight.norm(dim=1)
            gamma = norms[old].mean() / norms[new].mean().clamp(min=1e-12)
            fc.weight[new] *= gamma
        self.last_gamma = float(gamma)


# ---------------------------------------------------------------------------- CLIP zero-shot
class CLIPZeroShot(Head):
    """Frozen zero-shot head from CLIP text embeddings of the class names (CLIP, Radford et
    al. 2021): logits = 100 * cos(image feature, text embedding). The setting of MagMax /
    task_vectors, where only the image encoder is fine-tuned. Needs a ``CLIPModel`` and
    benchmark class names; ``templates`` default to CLIP's CIFAR templates."""

    scenarios = ("class", "domain")
    replaces_inference = False
    needs_features = False

    def __init__(self, templates=None, logit_scale: float = 100.0):
        self.templates, self.logit_scale = templates, logit_scale

    def bind(self, model, scenario):
        from cl_toolkit.models.base import CosineLinear
        super().bind(model, scenario)
        if not hasattr(model, "text_classifier"):
            raise TypeError("classifier='clip-zeroshot' needs a CLIP model (model='open_clip:ViT-B-16/openai')")
        bench = scenario.benchmark
        names = [bench.class_name(c).replace("_", " ") for c in range(self.num_classes)]
        if all(n.isdigit() for n in names):
            raise ValueError("the benchmark has no class names, which a zero-shot head needs")
        fc = model.get_head()._fc
        if not isinstance(fc, CosineLinear):
            raise TypeError("internal: the zero-shot head needs the cosine layer")
        with torch.no_grad():
            fc.weight.copy_(model.text_classifier(names, self.templates).to(fc.weight))
            fc.scale.fill_(self.logit_scale)
        fc.weight.requires_grad_(False)
        fc.scale.requires_grad_(False)


HEADS = {"ncm": NCM, "fecam": FeCAM, "lda": LDA, "slda": LDA, "klda": KLDA, "ranpac": RanPAC,
         "slca": SLCA, "wa": WA, "clip-zeroshot": CLIPZeroShot}


def get_head(name: str, **kw) -> Head:
    key = name.lower().replace("_", "-")
    if key not in HEADS:
        raise KeyError(f"Unknown classifier head {name!r}. Known: {sorted(HEADS)} (+ 'linear', 'cosine')")
    return HEADS[key](**kw)
