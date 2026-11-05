"""Prompt pools for frozen ViTs: L2P, DualPrompt, CODA-Prompt.

All three use *prefix-tuning*: learned key/value tokens are prepended inside
self-attention of selected layers, selected per-input from a pool using a
query = the frozen ViT's [CLS] feature. This matches DualPrompt and
CODA-Prompt; for L2P it is the "L2P++" variant (prefix instead of input
prompt tokens) used as a baseline in the CODA-Prompt paper, which performs
better than the original.

Usage::

    model = ViTModel("vit_base_patch16_224.augreg_in21k", freeze_backbone=True)
    model.attach_prompts(CODAPrompt(pool_size=100))
    cl_toolkit.run(..., model=model, strategy="naive")   # or strategy="coda-prompt"

Lengths follow the authors' code (CODA-Prompt repo, which re-implements all three):
a prompt of ``length`` tokens is split into ``length/2`` key and ``length/2`` value
prefix tokens. Defaults are that repo's CIFAR-100 / ImageNet-R settings.

A prompt module implements:
  * ``layers``: ViT block indices it writes prefixes for
  * ``forward(query) -> {layer: (pk, pv)}`` with pk/pv of shape (B, Lp, D)
  * ``begin_task(task_id, num_tasks)``: allocate / freeze per-task params
  * ``aux_loss``: extra loss from the last forward (key matching, ortho)
"""

from __future__ import annotations

import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Prefixes = Dict[int, Tuple[torch.Tensor, torch.Tensor]]


# ----------------------------------------------------------------------------
# Prefix injection into timm Attention
# ----------------------------------------------------------------------------

def _prefix_attn_forward(self, x, attn_mask=None, is_causal=False, **kw):
    prefix = getattr(self, "ctk_prefix", None)
    if prefix is None:
        return self._ctk_orig_forward(x, attn_mask=attn_mask, is_causal=is_causal, **kw)
    B, N, _ = x.shape
    H, hd = self.num_heads, self.head_dim
    qkv = self.qkv(x).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    pk, pv = prefix
    pk = pk.to(k.dtype).reshape(B, -1, H, hd).transpose(1, 2)
    pv = pv.to(v.dtype).reshape(B, -1, H, hd).transpose(1, 2)
    k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
    x = x.transpose(1, 2).reshape(B, N, -1)
    x = getattr(self, "norm", nn.Identity())(x)
    return self.proj_drop(self.proj(x))


def enable_prefix(vit: nn.Module, layers: List[int]) -> None:
    for i in layers:
        attn = vit.blocks[i].attn
        if not hasattr(attn, "_ctk_orig_forward"):
            attn._ctk_orig_forward = attn.forward
            attn.forward = types.MethodType(_prefix_attn_forward, attn)
            attn.ctk_prefix = None


def set_prefixes(vit: nn.Module, prefixes: Optional[Prefixes]) -> None:
    for i, blk in enumerate(vit.blocks):
        if hasattr(blk.attn, "_ctk_orig_forward"):
            blk.attn.ctk_prefix = None if prefixes is None else prefixes.get(i)


def _split_kv(p: torch.Tensor):
    """(..., L, D) -> two halves along L: keys, values."""
    h = p.size(-2) // 2
    return p[..., :h, :], p[..., h:, :]


def _even(name, L):
    if L % 2:
        raise ValueError(f"{name}={L} must be even (split into key/value halves)")
    return L


def _init(*shape):
    t = torch.empty(*shape)
    nn.init.uniform_(t)
    return nn.Parameter(t)


class PromptModule(nn.Module):
    layers: List[int] = []
    needs_query = True     # False: the model skips the extra frozen forward for the query

    def __init__(self):
        super().__init__()
        self.embed_dim: Optional[int] = None
        self.task_id = 0
        self._aux = None

    def bind(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def begin_task(self, task_id: int, num_tasks: int) -> None:
        self.task_id = task_id

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux if self._aux is not None else torch.zeros(())


# ----------------------------------------------------------------------------
# L2P (Wang et al., CVPR 2022) — prefix variant
# ----------------------------------------------------------------------------

class L2P(PromptModule):
    """Shared pool of ``pool_size`` prompts; top-``top_k`` by key/query cosine are used.

    Defaults = "L2P++" of the CODA-Prompt paper/repo (pool 30, length 20, top-5, layer 0).
    The original L2P (prompt-tuning, pool 10, length 5) is not implemented.
    """

    def __init__(self, pool_size: int = 30, length: int = 20, top_k: int = 5,
                 layers: List[int] = (0,), pull_coeff: float = 0.1):
        super().__init__()
        self.pool_size, self.length, self.top_k = pool_size, _even("length", length), top_k
        self.layers, self.pull_coeff = list(layers), pull_coeff

    def bind(self, d):
        super().bind(d)
        self.prompts = _init(self.pool_size, len(self.layers), self.length, d)
        self.keys = _init(self.pool_size, d)

    def forward(self, q: torch.Tensor) -> Prefixes:
        sim = F.normalize(q, dim=1) @ F.normalize(self.keys, dim=1).T          # (B, M)
        sel_sim, idx = sim.topk(self.top_k, dim=1)                              # (B, k)
        self._aux = self.pull_coeff * (1 - sel_sim).sum(1).mean()
        p = self.prompts[idx]                                                   # (B, k, nl, 2L, D)
        B = q.size(0)
        out = {}
        for j, layer in enumerate(self.layers):
            pk, pv = _split_kv(p[:, :, j])                                      # (B, k, L, D)
            out[layer] = (pk.reshape(B, -1, self.embed_dim), pv.reshape(B, -1, self.embed_dim))
        return out


# ----------------------------------------------------------------------------
# DualPrompt (Wang et al., ECCV 2022)
# ----------------------------------------------------------------------------

class DualPrompt(PromptModule):
    """G-prompt shared by all tasks (shallow layers) + one E-prompt per task (deeper layers).

    Training uses the true task's E-prompt; inference picks the E-prompt whose key
    is closest to the query (task-agnostic).
    """

    def __init__(self, g_layers=(0, 1), e_layers=(2, 3, 4), g_length: int = 6, e_length: int = 20,
                 pull_coeff: float = 1.0):
        super().__init__()
        self.g_layers, self.e_layers = list(g_layers), list(e_layers)
        self.layers = self.g_layers + self.e_layers
        self.g_length, self.e_length = _even("g_length", g_length), _even("e_length", e_length)
        self.pull_coeff = pull_coeff
        self.e_prompts = nn.ParameterList()
        self.e_keys = nn.ParameterList()

    def bind(self, d):
        super().bind(d)
        self.g_prompt = _init(len(self.g_layers), self.g_length, d)

    def begin_task(self, task_id, num_tasks):
        super().begin_task(task_id, num_tasks)
        dev = self.g_prompt.device
        while len(self.e_prompts) <= task_id:
            self.e_prompts.append(_init(len(self.e_layers), self.e_length, self.embed_dim).to(dev))
            self.e_keys.append(_init(self.embed_dim).to(dev))
        for i, (p, k) in enumerate(zip(self.e_prompts, self.e_keys)):
            p.requires_grad_(i == task_id)
            k.requires_grad_(i == task_id)

    def forward(self, q):
        B = q.size(0)
        keys = torch.stack(list(self.e_keys))                                   # (T, D)
        sim = F.normalize(q, dim=1) @ F.normalize(keys, dim=1).T                # (B, T)
        if self.training:
            idx = torch.full((B,), self.task_id, device=q.device, dtype=torch.long)
            self._aux = self.pull_coeff * (1 - sim[:, self.task_id]).mean()
        else:
            idx = sim.argmax(1)
            self._aux = None
        e = torch.stack(list(self.e_prompts))[idx]                              # (B, ne, 2L, D)
        out = {}
        for j, layer in enumerate(self.g_layers):
            pk, pv = _split_kv(self.g_prompt[j].unsqueeze(0).expand(B, -1, -1))
            out[layer] = (pk, pv)
        for j, layer in enumerate(self.e_layers):
            out[layer] = _split_kv(e[:, j])
        return out


# ----------------------------------------------------------------------------
# CODA-Prompt (Smith et al., CVPR 2023)
# ----------------------------------------------------------------------------

class CODAPrompt(PromptModule):
    """Prompts are an input-conditioned weighted sum of components.

    The pool is split evenly across tasks; at task t only the t-th slice of
    components/keys/attention vectors trains, older slices are frozen. New
    components are initialised orthogonal to existing ones (Gram-Schmidt).
    """

    def __init__(self, pool_size: int = 100, length: int = 8, layers=(0, 1, 2, 3, 4),
                 ortho_coeff: float = 0.0):
        super().__init__()
        self.pool_size, self.length, self.layers = pool_size, _even("length", length), list(layers)
        self.ortho_coeff = ortho_coeff
        self.P = nn.ParameterList()   # per task: (n, nl, L, D)
        self.K = nn.ParameterList()   # per task: (n, D)
        self.A = nn.ParameterList()   # per task: (n, D)

    def _orth_init(self, existing: Optional[torch.Tensor], n: int, d: int, dev) -> torch.Tensor:
        new = torch.randn(n, d, device=dev)
        if existing is not None and existing.numel():
            basis, _ = torch.linalg.qr(existing.T)           # (d, m)
            new = new - (new @ basis) @ basis.T
        q, _ = torch.linalg.qr(new.T)                          # orthonormal columns
        return q.T[:n].contiguous()

    def begin_task(self, task_id, num_tasks):
        super().begin_task(task_id, num_tasks)
        n = self.pool_size // num_tasks
        if n == 0:
            raise ValueError(f"pool_size={self.pool_size} < num_tasks={num_tasks}")
        dev = next(self.parameters()).device if len(self.P) else torch.device("cpu")
        d, nl, L = self.embed_dim, len(self.layers), self.length
        while len(self.P) <= task_id:
            old = lambda pl: torch.cat([p.detach().reshape(p.size(0), -1) for p in pl]) if len(pl) else None
            self.K.append(nn.Parameter(self._orth_init(old(self.K), n, d, dev)))
            self.A.append(nn.Parameter(self._orth_init(old(self.A), n, d, dev)))
            p = torch.stack([self._orth_init(None if not len(self.P) else
                                             torch.cat([q[:, j].reshape(q.size(0), -1) for q in self.P]).detach(),
                                             n, L * d, dev) for j in range(nl)], 1)
            self.P.append(nn.Parameter(p.reshape(n, nl, L, d)))
        for i in range(len(self.P)):
            for pl in (self.P, self.K, self.A):
                pl[i].requires_grad_(i == task_id)

    def forward(self, q):
        t = self.task_id + 1 if self.training else len(self.P)
        K = torch.cat(list(self.K)[:t])                                         # (M, D)
        A = torch.cat(list(self.A)[:t])
        P = torch.cat(list(self.P)[:t])                                         # (M, nl, L, D)
        qa = q.unsqueeze(1) * A.unsqueeze(0)                                    # (B, M, D)
        alpha = (F.normalize(qa, dim=2) * F.normalize(K, dim=1).unsqueeze(0)).sum(2)  # (B, M)
        prompt = torch.einsum("bm,mnld->bnld", alpha, P)                       # (B, nl, L, D)
        self._aux = None
        if self.training and self.ortho_coeff > 0:
            def ortho(x):
                x = x.reshape(x.size(0), -1)
                g = x @ x.T
                return ((g - torch.eye(g.size(0), device=g.device)) ** 2).mean()
            self._aux = self.ortho_coeff * (ortho(K) + ortho(A) + ortho(P))
        return {layer: _split_kv(prompt[:, j]) for j, layer in enumerate(self.layers)}


# ----------------------------------------------------------------------------
# Per-task prompt sets, selected from outside (for cl_toolkit.routing)
# ----------------------------------------------------------------------------

class TaskPrompts(PromptModule):
    """One prefix-prompt set per task; *which* set each sample uses is decided
    externally: ``ids`` (per-sample tensor) or ``train_id`` (int, training). With
    neither set, no prompts are applied (the plain PTM, used as router query).

    Used by :class:`cl_toolkit.routing.PromptExperts`.
    """

    needs_query = False

    def __init__(self, length: int = 20, layers=(0, 1, 2, 3, 4)):
        super().__init__()
        self.length, self.layers = _even("length", length), list(layers)
        self.prompts = nn.ParameterList()
        self.ids: Optional[torch.Tensor] = None
        self.train_id: Optional[int] = None

    def add(self, task_id: int) -> nn.Parameter:
        dev = self.prompts[0].device if len(self.prompts) else torch.device("cpu")
        while len(self.prompts) <= task_id:
            self.prompts.append(_init(len(self.layers), self.length, self.embed_dim).to(dev))
        for i, p in enumerate(self.prompts):
            p.requires_grad_(i == task_id)
        return self.prompts[task_id]

    def forward(self, q, batch_size: Optional[int] = None) -> Prefixes:
        if self.ids is None and self.train_id is None:
            self._aux = None
            return {}
        B = batch_size if q is None else q.size(0)
        if self.ids is not None:
            P = torch.stack(list(self.prompts))[self.ids]                       # (B, nl, 2L, D)
        else:
            P = self.prompts[self.train_id].unsqueeze(0).expand(B, -1, -1, -1)
        self._aux = None
        return {layer: _split_kv(P[:, j]) for j, layer in enumerate(self.layers)}


PROMPTS = {"l2p": L2P, "dualprompt": DualPrompt, "coda-prompt": CODAPrompt, "coda": CODAPrompt}
