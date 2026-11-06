"""Routing: one expert per task, a router picks the expert per input at test time.

Experts (trained only on their own task, frozen afterwards):

    LoRAExperts         one LoRA adapter per task                     (peft)
    AdaptFormerExperts  one AdaptFormer bottleneck adapter per task
    PromptExperts       one prefix-prompt set per task                (timm ViT)

Routers (see each docstring for the paper). All route from the *query*: the
plain pre-trained model's feature (no expert applied), which never changes:

    KeyQuery     learned key per task, cosine to query           (L2P / DualPrompt, Wang et al. 2022)
    SPrompts     k-means centroids per task, nearest centroid    (S-Prompts, Wang et al., NeurIPS 2022)
    HiDe         class classifier on Gaussian-sampled query      (HiDe-Prompt, Wang et al., NeurIPS 2023)
                 features of all seen classes -> owning task
    ExpertGate   one autoencoder per task, min reconstruction    (Expert Gate, Aljundi et al., CVPR 2017)
    HeadRouter   any cl_toolkit.heads head (LDA, FeCAM, ...) on queries -> class -> owning task

Combine with :class:`cl_toolkit.strategies.RoutedExperts`, or
``clt.run(..., strategy="routed", strategy_kwargs={"experts": "lora", "router": "hide"})``.

Class-level routers (HiDe, HeadRouter) map a predicted class to its task, so they
need disjoint classes per task (Class-IL). Task-level routers also work in Domain-IL.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cl_toolkit import batch as B
import torch.nn as nn
import torch.nn.functional as F

from cl_toolkit import adapters


# ============================================================================ experts
class ExpertPool:
    """Interface: add / train-mode / plain-PTM mode / per-sample routed encoding."""

    def add(self, model, task_id: int) -> None: ...
    def use_train(self, model, task_id: int) -> None: ...
    def use_none(self, model) -> None: ...
    def freeze(self, model, task_id: int) -> None: ...

    def encode_with(self, model, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Features of each sample under its expert ``ids[i]`` (grouped forward)."""
        out = None
        for e in ids.unique().tolist():
            m = ids == e
            self._activate(model, int(e))
            f = model.encode(B.index(x, m))
            if out is None:
                out = f.new_empty((B.size(x), f.size(1)))
            out[m] = f
        return out

    def _activate(self, model, e: int) -> None:
        raise NotImplementedError

    def __repr__(self):
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items() if isinstance(v, (int, float, str)))
        return f"{type(self).__name__}({args})"


class LoRAExperts(ExpertPool):
    """``targets`` / ``layers``: where LoRA goes (see :func:`cl_toolkit.adapters.resolve_lora_targets`)."""

    def __init__(self, rank: int = 8, alpha: float = 16.0, targets=None, layers=None, target_modules=None):
        self.rank, self.alpha = rank, alpha
        self.targets, self.layers = (targets if targets is not None else target_modules), layers

    def add(self, model, task_id):
        adapters.add_lora(model, f"expert_{task_id}", self.rank, self.alpha, targets=self.targets, layers=self.layers)

    def _activate(self, model, e, trainable=()):
        # enable_adapter_layers() marks every active adapter trainable, so set the flags explicitly
        model._backbone.enable_adapter_layers()
        adapters.activate_lora(model, [f"expert_{e}"], trainable=list(trainable))

    def use_train(self, model, task_id):
        self._activate(model, task_id, trainable=[f"expert_{task_id}"])

    def use_none(self, model):
        if adapters.lora_names(model):
            model._backbone.disable_adapter_layers()

    def freeze(self, model, task_id):
        adapters.freeze_lora(model, f"expert_{task_id}")


class AdaptFormerExperts(ExpertPool):
    def __init__(self, bottleneck: int = 64, scale: float = 0.1):
        self.bottleneck, self.scale = bottleneck, scale

    def add(self, model, task_id):
        adapters.add_adaptformer(model, self.bottleneck, self.scale, name=f"expert_{task_id}")
        for p in adapters.adaptformer_params(model, f"expert_{task_id}"):
            p.requires_grad_(True)

    def _activate(self, model, e):
        adapters.set_adaptformer_active(model, [f"expert_{e}"])

    use_train = _activate

    def use_none(self, model):
        adapters.set_adaptformer_active(model, [])

    def freeze(self, model, task_id):
        for p in adapters.adaptformer_params(model, f"expert_{task_id}"):
            p.requires_grad_(False)


class PromptExperts(ExpertPool):
    """Per-task prefix prompts; routed samples are processed in a *single* forward."""

    def __init__(self, length: int = 20, layers=(0, 1, 2, 3, 4)):
        self.length, self.layers = length, tuple(layers)
        self._tp = None

    def add(self, model, task_id):
        from cl_toolkit.prompts import TaskPrompts
        if self._tp is None:
            if not hasattr(model, "attach_prompts"):
                raise TypeError(f"prompt experts insert prefix prompts into timm ViT attention; "
                                f"{type(model).__name__} is not a timm ViT (use experts='lora')")
            self._tp = TaskPrompts(self.length, self.layers)
            model.attach_prompts(self._tp)
        self._tp.add(task_id)
        self._tp.to(model._device)

    def use_train(self, model, task_id):
        self._tp.ids, self._tp.train_id = None, task_id

    def use_none(self, model):
        self._tp.ids = self._tp.train_id = None

    def freeze(self, model, task_id):
        self._tp.prompts[task_id].requires_grad_(False)

    def encode_with(self, model, x, ids):
        self._tp.ids, self._tp.train_id = ids, None
        try:
            return model.encode(x)
        finally:
            self._tp.ids = None


EXPERTS = {"lora": LoRAExperts, "adaptformer": AdaptFormerExperts, "prompt": PromptExperts}


# ============================================================================ routers
class Router:
    scenarios = ("class", "domain")
    trainable = False

    def bind(self, scenario) -> None:
        self.scenario = scenario

    def begin_task(self, task_id: int, dim: int, device) -> None: ...

    def params(self, task_id: int) -> List[nn.Parameter]:
        return []

    def aux_loss(self, q: torch.Tensor, task_id: int) -> torch.Tensor:
        return q.new_zeros(())

    def fit(self, task_id: int, q: torch.Tensor, y: torch.Tensor) -> None:
        """Absorb the plain-PTM features ``q`` (and global labels) of task ``task_id``."""

    def route(self, q: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B,) task/expert ids."""
        raise NotImplementedError

    def __repr__(self):
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items()
                         if not k.startswith("_") and isinstance(v, (int, float, str)) and k != "scenario")
        return f"{type(self).__name__}({args})"


class KeyQuery(Router):
    """A learned key per task, pulled towards that task's queries during training
    (``aux_loss``); route = argmax cosine(query, key). The selection rule of
    L2P / DualPrompt (Wang et al., CVPR / ECCV 2022)."""

    trainable = True

    def __init__(self, pull_coeff: float = 1.0):
        self.pull_coeff = pull_coeff
        self._keys = nn.ParameterList()

    def begin_task(self, task_id, dim, device):
        while len(self._keys) <= task_id:
            self._keys.append(nn.Parameter(torch.randn(dim, device=device) * 0.02))
        for i, k in enumerate(self._keys):
            k.requires_grad_(i == task_id)

    def params(self, task_id):
        return [self._keys[task_id]]

    def aux_loss(self, q, task_id):
        return self.pull_coeff * (1 - F.cosine_similarity(q.float(), self._keys[task_id].unsqueeze(0))).mean()

    def route(self, q):
        K = F.normalize(torch.stack(list(self._keys)), dim=1)
        return (F.normalize(q.float(), dim=1) @ K.T).argmax(1)


def _kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    k = min(k, len(x))
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


class SPrompts(Router):
    """K-means centroids of each task's (normalised) queries; route to the task owning
    the nearest centroid (S-Prompts, Wang et al., NeurIPS 2022; paper uses K=5)."""

    def __init__(self, k: int = 5, seed: int = 0):
        self.k, self.seed = k, seed
        self._centroids: List[torch.Tensor] = []
        self._owner: List[torch.Tensor] = []

    def fit(self, task_id, q, y):
        c = _kmeans(F.normalize(q.float(), dim=1), self.k, seed=self.seed + task_id)
        self._centroids.append(c)
        self._owner.append(torch.full((len(c),), task_id, device=c.device))

    def route(self, q):
        C, own = torch.cat(self._centroids), torch.cat(self._owner)
        return own[torch.cdist(F.normalize(q.float(), dim=1), C).argmin(1)]


class _ClassToTask(Router):
    scenarios = ("class",)

    def _task_of(self) -> torch.Tensor:
        t_of = torch.full((self.scenario.benchmark.total_classes,), -1, dtype=torch.long)
        for t in self.scenario.tasks:
            t_of[t.classes] = t.task_id
        return t_of


class HiDe(_ClassToTask):
    """Task-identity inference of HiDe-Prompt (Wang et al., NeurIPS 2023): keep a
    Gaussian (mean, covariance) of the plain-PTM features of every class; after each
    task train a linear classifier over all seen classes on features sampled from
    those Gaussians; route to the task owning the predicted class."""

    def __init__(self, samples_per_class: int = 256, epochs: int = 20, lr: float = 1e-3,
                 shrinkage: float = 1e-4, seed: int = 0):
        self.samples_per_class, self.epochs, self.lr, self.shrinkage, self.seed = \
            samples_per_class, epochs, lr, shrinkage, seed
        self._stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._clf = None

    def fit(self, task_id, q, y):
        q = q.float()
        D = q.size(1)
        for c in y.unique().tolist():
            qc = q[y == c].double()
            cov = (torch.cov(qc.T) if len(qc) > 1 else torch.zeros(D, D, dtype=qc.dtype, device=qc.device))
            cov = cov + self.shrinkage * torch.eye(D, dtype=cov.dtype, device=cov.device)
            self._stats[c] = (qc.mean(0), torch.linalg.cholesky(cov))
        g = torch.Generator(device=q.device).manual_seed(self.seed + task_id)
        xs, ys = [], []
        for c, (mu, L) in self._stats.items():
            z = torch.randn(self.samples_per_class, D, generator=g, device=q.device, dtype=mu.dtype)
            xs.append((mu + z @ L.T).float())
            ys.append(torch.full((self.samples_per_class,), c, device=q.device))
        x, yy = torch.cat(xs), torch.cat(ys)
        C = self.scenario.benchmark.total_classes
        self._clf = nn.Linear(D, C).to(q.device)
        seen = torch.tensor(sorted(self._stats), device=q.device)
        opt = torch.optim.Adam(self._clf.parameters(), lr=self.lr)
        with torch.enable_grad():
            for _ in range(self.epochs):
                perm = torch.randperm(len(x), device=x.device)
                for i in range(0, len(x), 256):
                    b = perm[i:i + 256]
                    logits = self._clf(x[b])[:, seen]
                    loss = F.cross_entropy(logits, torch.searchsorted(seen, yy[b]))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
        self._seen, self._t_of = seen, self._task_of().to(q.device)

    @torch.no_grad()
    def route(self, q):
        cls = self._seen[self._clf(q.float())[:, self._seen].argmax(1)]
        return self._t_of[cls]


class ExpertGate(Router):
    """One undercomplete autoencoder per task on its (normalised) queries; route to the
    task whose autoencoder reconstructs the input best (Aljundi et al., CVPR 2017)."""

    def __init__(self, hidden: int = 128, epochs: int = 50, lr: float = 1e-3, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        self._aes = nn.ModuleList()

    def fit(self, task_id, q, y):
        torch.manual_seed(self.seed + task_id)
        x = F.normalize(q.float(), dim=1)
        D = x.size(1)
        ae = nn.Sequential(nn.Linear(D, self.hidden), nn.ReLU(), nn.Linear(self.hidden, D)).to(x.device)
        opt = torch.optim.Adam(ae.parameters(), lr=self.lr)
        with torch.enable_grad():
            for _ in range(self.epochs):
                perm = torch.randperm(len(x), device=x.device)
                for i in range(0, len(x), 256):
                    b = x[perm[i:i + 256]]
                    loss = F.mse_loss(ae(b), b)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
        ae.eval().requires_grad_(False)
        self._aes.append(ae)

    @torch.no_grad()
    def route(self, q):
        x = F.normalize(q.float(), dim=1)
        err = torch.stack([((ae(x) - x) ** 2).sum(1) for ae in self._aes], 1)
        return err.argmin(1)


class HeadRouter(_ClassToTask):
    """Any :mod:`cl_toolkit.heads` statistical head (NCM, LDA, KLDA, FeCAM, RanPAC)
    fitted on plain-PTM queries: predicted class -> owning task."""

    def __init__(self, head="lda", **head_kw):
        from cl_toolkit.heads import Head, get_head
        self.head = head if isinstance(head, Head) else get_head(head, **head_kw)

    def bind(self, scenario):
        super().bind(scenario)
        self.head.bind(None, scenario)

    def fit(self, task_id, q, y):
        self.head.update(q.float(), y, task_id)
        self._t_of = self._task_of().to(q.device)

    @torch.no_grad()
    def route(self, q):
        return self._t_of[self.head.logits(q.float()).argmax(1)]

    def __repr__(self):
        return f"HeadRouter({self.head!r})"


ROUTERS = {"keyquery": KeyQuery, "key-query": KeyQuery, "sprompts": SPrompts, "s-prompts": SPrompts,
           "hide": HiDe, "expertgate": ExpertGate, "expert-gate": ExpertGate, "head": HeadRouter}


def get_router(name, **kw) -> Router:
    if isinstance(name, Router):
        return name
    key = name.lower().replace("_", "-")
    if key in ROUTERS:
        return ROUTERS[key](**kw)
    from cl_toolkit.heads import HEADS
    if key in HEADS:                                      # e.g. router="lda"
        return HeadRouter(key, **kw)
    raise KeyError(f"Unknown router {name!r}. Known: {sorted(ROUTERS)} or a head name {sorted(HEADS)}")


def get_experts(name, **kw) -> ExpertPool:
    if isinstance(name, ExpertPool):
        return name
    if name not in EXPERTS:
        raise KeyError(f"Unknown experts {name!r}. Known: {sorted(EXPERTS)}")
    return EXPERTS[name](**kw)
