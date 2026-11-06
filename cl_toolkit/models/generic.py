"""Use *any* model with the toolkit.

``wrap(module)`` turns an encoder (``inputs -> features``) into a CLModel: every
strategy, head, adapter, router and merging method then works with it. Inputs
can be tensors, dicts (tokenised text) or tuples (see :mod:`cl_toolkit.batch`).

``HFModel("bert-base-uncased")`` (or ``model="hf:<name>"`` in ``clt.run``) loads a
Hugging Face encoder *or* decoder LLM as a feature extractor, with its tokenizer
as the data transform.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn

from .base import CLModel


def _infer_dim(module: nn.Module) -> Optional[int]:
    for attr in ("num_features", "embed_dim", "hidden_size", "output_dim", "out_features"):
        v = getattr(module, attr, None)
        if isinstance(v, int):
            return v
    cfg = getattr(module, "config", None)
    for attr in ("hidden_size", "d_model", "n_embd", "dim"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            return v
    return None


def _pool_output(out):
    """Common encoder outputs -> (B, D) features."""
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    if torch.is_tensor(out) and out.dim() == 3:          # (B, tokens, D) -> first token
        out = out[:, 0]
    return out


class FeatureModel(CLModel):
    """Any encoder as a CLModel.

    Args:
        module: the encoder (stored as the backbone).
        embed_dim: feature size; inferred from common attributes when omitted.
        encode: ``f(module, x) -> (B, D)``; default calls ``module(x)`` (or ``module(**x)``
            for dict inputs) and pools common output types.
        lora_targets: default LoRA placement (names / regex / "all-linear"); default "all-linear".
        transforms: optional ``(train_tf, test_tf)`` used by named datasets.
    """

    def __init__(self, module: nn.Module, embed_dim: Optional[int] = None, encode: Optional[Callable] = None,
                 lora_targets: Optional[List[str]] = None, transforms=None, freeze_backbone: bool = False,
                 device=None, name: Optional[str] = None):
        super().__init__(device)
        self._backbone = module.to(self._device)
        self._encode_fn = encode
        self._embed_dim = embed_dim or _infer_dim(module)
        if self._embed_dim is None:
            raise ValueError("Could not infer embed_dim; pass wrap(module, embed_dim=...)")
        self._lora_targets = lora_targets
        self._transforms = transforms
        self.model_name = name or type(module).__name__
        if freeze_backbone:
            self.freeze_backbone()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _encode(self, x):
        if self._encode_fn is not None:
            return self._encode_fn(self._backbone, x)
        out = self._backbone(**x) if isinstance(x, dict) else self._backbone(x)
        return _pool_output(out)

    def _default_lora_targets(self):
        return self._lora_targets or "all-linear"

    def transforms(self):
        if self._transforms is None:
            raise AttributeError("this model has no transforms; pass wrap(..., transforms=(train, test))")
        return self._transforms


def mlp(in_dim: int = 32, hidden: int = 128, device=None) -> FeatureModel:
    """A small, randomly initialised MLP encoder for vector inputs (``model="mlp"``; pairs with
    ``data="toy"``). Untrained, so ``clt.run`` trains it by default."""
    net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden))
    return FeatureModel(net, embed_dim=hidden, device=device, name="mlp")


def wrap(module: nn.Module, embed_dim: Optional[int] = None, **kw) -> FeatureModel:
    """``clt.models.wrap(my_encoder)`` -> a CLModel usable everywhere in the toolkit."""
    return FeatureModel(module, embed_dim, **kw)


class HFModel(FeatureModel):
    """A Hugging Face model (encoder like BERT, or decoder LLM like Qwen/Llama) as a
    feature extractor. Its tokenizer is the data transform (``model.transforms()``).

    pooling: "cls" (first token), "mean" (masked mean), "last" (last non-pad token,
    the usual choice for decoder LLMs). Default: "last" for decoders, else "cls".
    """

    LORA_TARGETS = ["query", "value", "q_proj", "v_proj", "q_lin", "v_lin", "query_key_value", "c_attn"]

    def __init__(self, name: str, pooling: Optional[str] = None, max_length: int = 128, freeze_backbone=True,
                 device=None, dtype=None, **from_pretrained_kw):
        from transformers import AutoModel, AutoTokenizer
        backbone = AutoModel.from_pretrained(name, dtype=dtype, **from_pretrained_kw) if dtype else \
            AutoModel.from_pretrained(name, **from_pretrained_kw)
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        is_decoder = bool(getattr(backbone.config, "is_decoder", False)) or \
            getattr(backbone.config, "model_type", "") in ("llama", "qwen2", "qwen3", "mistral", "gpt2", "phi3", "gemma")
        self.pooling = pooling or ("last" if is_decoder else "cls")
        self.max_length, self.tokenizer = max_length, tok

        def tokenize(text):
            enc = tok(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

        super().__init__(backbone, embed_dim=_infer_dim(backbone), encode=self._hf_encode,
                         lora_targets=self.LORA_TARGETS, transforms=(tokenize, tokenize),
                         freeze_backbone=freeze_backbone, device=device, name=name)

    def _hf_encode(self, module, x):
        h = module(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
        if self.pooling == "mean":
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        if self.pooling == "last":
            pos = torch.arange(h.size(1), device=h.device)          # last non-pad token, any padding side
            idx = (x["attention_mask"].to(pos.dtype) * (pos + 1)).argmax(1)
            return h[torch.arange(h.size(0), device=h.device), idx]
        return h[:, 0]
