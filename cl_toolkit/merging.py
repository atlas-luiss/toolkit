"""Model merging on task vectors (tau_t = theta_t - theta_0).

Every method takes a list of task vectors ``[{name: tensor}, ...]`` and returns one merged
task vector; ``apply(base, merged)`` adds it to the pre-trained weights. In CL, use
:class:`cl_toolkit.strategies.MergeAfterTask` (or :class:`~cl_toolkit.strategies.ModelAveraging`
for CoMA / CoFiMA, which average models recursively).

==================  ==============================================================  ======================
method              rule                                                            default (source)
==================  ==============================================================  ======================
``average``         mean of task vectors (weight averaging)                         —
``task_arithmetic`` lam * sum (Ilharco et al., ICLR 2023)                           lam 0.3 (paper: tuned)
``ties``            trim to top-k% *of each whole task vector*, elect sign by       k 20%, lam 1
                    summed mass, disjoint mean (Yadav et al., NeurIPS 2023)          (authors' notebook)
``dare``            drop with prob. p, rescale by 1/(1-p), then a base method       p 0.9, base TA
                    (Yu et al., ICML 2024)                                          (paper: tuned)
``magmax``          per-coordinate max magnitude (Marczak et al., ECCV 2024)         lam 0.5 (paper)
``breadcrumbs``     per layer, drop the smallest beta and the largest (1-gamma)      beta .90, gamma .992
                    of |tau|, then lam * sum (Davari & Belilovsky, ECCV 2024)        (paper, ViT-B/16);
                                                                                    lam 0.3 (paper: tuned)
``consensus``       TALL masks m_t = |tau_t| >= l |tau_MTL - tau_t|; keep weights    k 2, l 0.4, base TA
                    selected by >= k tasks (Wang et al., ICML 2024)                  (paper: l, lam tuned)
``tsv``             per matrix: top 1/T singular components of each task,           alpha 1.0 (paper)
                    orthogonalised U, V (Procrustes) (Gargiulo et al., CVPR 2025)
``iso_c``           per matrix: SVD of the sum, all singular values -> their mean   alpha 1.0
                    (Marczak et al., ICML 2025)                                     (paper: tuned)
``iso_cts``         Iso-C on a common subspace (k/r = 0.8) + task-specific          alpha 1.0
                    subspaces, orthogonalised (Marczak et al., ICML 2025)            (paper: tuned)
==================  ==============================================================  ======================

For the SVD methods (``tsv``, ``iso_c``, ``iso_cts``), parameters that are not 2-D matrices
(biases, norms, conv kernels) are averaged, as in those papers. Where a paper tunes its
scaling coefficient on held-out validation data, the default here is fixed: tune it on a
validation split of the *training* data, never on the test set.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import zlib

import torch

TV = Dict[str, torch.Tensor]


def task_vector(finetuned: TV, base: TV) -> TV:
    return {k: finetuned[k] - base[k] for k in base}


def apply(base: TV, merged_tv: TV, lam: float = 1.0) -> TV:
    return {k: base[k] + lam * merged_tv[k].to(base[k]) for k in base}


def _stack(tvs: List[TV], k: str) -> torch.Tensor:
    return torch.stack([tv[k].float() for tv in tvs])


# ---------------------------------------------------------------------------- element-wise
def average(tvs: List[TV]) -> TV:
    return {k: _stack(tvs, k).mean(0) for k in tvs[0]}


def task_arithmetic(tvs: List[TV], lam: float = 0.3) -> TV:
    return {k: lam * _stack(tvs, k).sum(0) for k in tvs[0]}


def magmax(tvs: List[TV], lam: float = 0.5) -> TV:
    out = {}
    for k in tvs[0]:
        s = _stack(tvs, k)
        idx = s.abs().argmax(0, keepdim=True)
        out[k] = lam * s.gather(0, idx).squeeze(0)
    return out


def _kth_largest_abs(values: torch.Tensor, k: int) -> torch.Tensor:
    flat = values.abs().flatten()
    return flat.kthvalue(flat.numel() - k + 1).values


def _trim_global(tv: TV, density: float) -> TV:
    """Keep the top ``density`` fraction of |tau| over *all* parameters of one task vector."""
    if density >= 1:
        return {k: v.float() for k, v in tv.items()}
    flat = torch.cat([v.float().flatten() for v in tv.values()])
    thr = _kth_largest_abs(flat, max(int(density * flat.numel()), 1))
    return {k: v.float() * (v.float().abs() >= thr) for k, v in tv.items()}


def _trim_tensor(t: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1:
        return t
    thr = _kth_largest_abs(t, max(int(density * t.numel()), 1))
    return t * (t.abs() >= thr)


def ties(tvs: List[TV], density: float = 0.2, lam: float = 1.0, scope: str = "global") -> TV:
    """TIES-Merging. ``scope="global"`` trims over each whole task vector (authors' code);
    ``"tensor"`` trims each parameter tensor separately."""
    if scope == "global":
        trimmed = [_trim_global(tv, density) for tv in tvs]
    elif scope == "tensor":
        trimmed = [{k: _trim_tensor(v.float(), density) for k, v in tv.items()} for tv in tvs]
    else:
        raise ValueError("scope must be 'global' or 'tensor'")
    # sign election by total mass; zero signs -> the majority sign over all parameters
    signs = {k: torch.sign(_stack(trimmed, k).sum(0)) for k in tvs[0]}
    total = sum(float(s.sum()) for s in signs.values())
    majority = 1.0 if total >= 0 else -1.0
    out = {}
    for k in tvs[0]:
        s = _stack(trimmed, k)
        sign = signs[k].masked_fill(signs[k] == 0, majority)
        agree = (torch.sign(s) == sign) & (s != 0)
        out[k] = lam * (s * agree).sum(0) / agree.sum(0).clamp(min=1)
    return out


def dare(tvs: List[TV], p: float = 0.9, base: Callable = None, seed: int = 0, **base_kw) -> TV:
    """DARE: drop each delta with probability ``p``, rescale survivors by 1/(1-p), then ``base``
    (default: Task Arithmetic; pass ``base=ties`` for DARE-TIES)."""
    base = base or task_arithmetic
    dropped = []
    for t, tv in enumerate(tvs):
        d = {}
        for k, v in tv.items():
            # one stream per (seed, task, parameter name): masks do not depend on dict order
            g = torch.Generator().manual_seed(zlib.crc32(f"{seed}/{t}/{k}".encode()))
            keep = (torch.rand(v.shape, generator=g) >= p).to(v.device)
            d[k] = v.float() * keep / (1 - p)
        dropped.append(d)
    return base(dropped, **base_kw)


def breadcrumbs(tvs: List[TV], beta: float = 0.90, gamma: float = 0.992, lam: float = 0.3) -> TV:
    """Model Breadcrumbs: per layer and task, zero the smallest ``beta`` fraction and the largest
    ``1 - gamma`` fraction of |tau|, then ``lam * sum``."""
    if not 0 <= beta < gamma <= 1:
        raise ValueError("need 0 <= beta < gamma <= 1")
    out = {}
    for k in tvs[0]:
        s = _stack(tvs, k)
        flat = s.abs().reshape(len(tvs), -1)
        n = flat.size(1)
        order = flat.argsort(1).argsort(1)                     # rank of each entry within its layer
        keep = (order >= int(beta * n)) & (order < max(int(gamma * n), int(beta * n) + 1))
        out[k] = lam * (s * keep.reshape(s.shape)).sum(0)
    return out


def tall_masks(tvs: List[TV], lam: float = 0.4) -> List[Dict[str, torch.Tensor]]:
    """TALL masks (Wang et al., ICML 2024, Eq. 5): m_t = |tau_t| >= lam * |tau_MTL - tau_t|,
    with tau_MTL the sum of all task vectors."""
    mtl = {k: _stack(tvs, k).sum(0) for k in tvs[0]}
    return [{k: tv[k].float().abs() >= lam * (mtl[k] - tv[k].float()).abs() for k in tv} for tv in tvs]


def consensus(tvs: List[TV], k: int = 2, tall_lambda: float = 0.4, base: Callable = None, **base_kw) -> TV:
    """Consensus merging: keep only weights selected by at least ``k`` TALL masks, applied to the
    multi-task vector of ``base`` (default Task Arithmetic; the paper uses k=2 for TA, k=1 for TIES)."""
    base = base or task_arithmetic
    masks = tall_masks(tvs, tall_lambda)
    merged = base(tvs, **base_kw)
    return {n: merged[n] * (torch.stack([m[n] for m in masks]).sum(0) >= k) for n in merged}


# ---------------------------------------------------------------------------- SVD-based (per matrix)
def _is_matrix(t: torch.Tensor) -> bool:
    return t.dim() == 2 and min(t.shape) > 1


def _svd(X: torch.Tensor):
    """Reduced SVD that survives ill-conditioned inputs (LAPACK's default driver can fail to
    converge): retry in float64, then with a 1e-12 relative jitter."""
    try:
        return torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        Xd = X.double()
        try:
            U, S, Vh = torch.linalg.svd(Xd, full_matrices=False)
        except RuntimeError:
            jitter = 1e-12 * Xd.abs().max().clamp(min=1e-30) * torch.randn_like(Xd)
            U, S, Vh = torch.linalg.svd(Xd + jitter, full_matrices=False)
        return U.to(X.dtype), S.to(X.dtype), Vh.to(X.dtype)


def _orthogonalise(X: torch.Tensor) -> torch.Tensor:
    """Closest matrix with orthonormal columns (orthogonal Procrustes): P Q^T from X = P D Q^T."""
    P, _, Qh = _svd(X)
    return P @ Qh


def tsv(tvs: List[TV], alpha: float = 1.0) -> TV:
    """Task Singular Vectors merging (TSV-M). Per matrix: keep each task's top ``r // T``
    singular components, orthogonalise the concatenated U and V, reconstruct."""
    T, out = len(tvs), {}
    for k in tvs[0]:
        if not _is_matrix(tvs[0][k]):
            out[k] = _stack(tvs, k).mean(0)
            continue
        r = min(tvs[0][k].shape)
        kk = max(r // T, 1)
        Us, Ss, Vs = [], [], []
        for tv in tvs:
            U, S, Vh = _svd(tv[k].float())
            Us.append(U[:, :kk]), Ss.append(S[:kk]), Vs.append(Vh[:kk].T)
        U, S, V = torch.cat(Us, 1), torch.cat(Ss), torch.cat(Vs, 1)
        out[k] = alpha * (_orthogonalise(U) * S) @ _orthogonalise(V).T
    return out


def iso_c(tvs: List[TV], alpha: float = 1.0) -> TV:
    """Iso-C: per matrix, SVD of the summed task matrices; replace every singular value by their mean."""
    out = {}
    for k in tvs[0]:
        if not _is_matrix(tvs[0][k]):
            out[k] = _stack(tvs, k).mean(0)
            continue
        U, S, Vh = _svd(_stack(tvs, k).sum(0))
        out[k] = alpha * S.mean() * (U @ Vh)
    return out


def iso_cts(tvs: List[TV], alpha: float = 1.0, common_fraction: float = 0.8) -> TV:
    """Iso-CTS: top-k (k = common_fraction * r) directions of the sum + each task's top
    (r - k) / T directions orthogonal to them; orthogonalise; isotropic spectrum (their mean)."""
    T, out = len(tvs), {}
    for k in tvs[0]:
        if not _is_matrix(tvs[0][k]):
            out[k] = _stack(tvs, k).mean(0)
            continue
        r = min(tvs[0][k].shape)
        kc = min(max(int(round(common_fraction * r)), 1), r)
        s = (r - kc) // T
        U, S, Vh = _svd(_stack(tvs, k).sum(0))
        Uc, Vc, sig = U[:, :kc], Vh[:kc].T, [S[:kc]]
        Us, Vs = [Uc], [Vc]
        if s > 0:
            for tv in tvs:
                D = tv[k].float()
                D = D - Uc @ (Uc.T @ D)                         # project out the common subspace
                Ut, St, Vth = _svd(D)
                Us.append(Ut[:, :s]), Vs.append(Vth[:s].T), sig.append(St[:s])
        sig_bar = torch.cat(sig).sum() / (kc + T * s)
        out[k] = alpha * sig_bar * (_orthogonalise(torch.cat(Us, 1)) @ _orthogonalise(torch.cat(Vs, 1)).T)
    return out


METHODS: Dict[str, Callable] = {
    "average": average, "task_arithmetic": task_arithmetic, "ta": task_arithmetic, "ties": ties, "dare": dare,
    "magmax": magmax, "breadcrumbs": breadcrumbs, "consensus": consensus, "tsv": tsv,
    "iso_c": iso_c, "iso-c": iso_c, "iso_cts": iso_cts, "iso-cts": iso_cts,
}


def merge(tvs: List[TV], method: str = "ties", **kw) -> TV:
    if method not in METHODS:
        raise KeyError(f"Unknown merging method {method!r}. Known: {sorted(METHODS)}")
    return METHODS[method](tvs, **kw)


def similarity(a: TV, b: TV) -> float:
    """Cosine similarity between two task vectors (flattened)."""
    va = torch.cat([a[k].flatten().float() for k in a])
    vb = torch.cat([b[k].flatten().float() for k in a])
    return float(torch.nn.functional.cosine_similarity(va, vb, dim=0))
