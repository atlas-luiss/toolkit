"""HAM — Hierarchical Adapter Merging: one LoRA per task, similarity-based grouping, DARE-TIES merge."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from peft import LoraConfig
except ImportError:
    raise ImportError("peft is required: pip install peft")

from ctk.strategies.base import Strategy
from ctk.utils.adapter_ops import calculate_adapter_similarity
from ctk.utils.lora_merge import (
    build_adapter_from_weights,
    concat_lora_adapters,
    merge_lora_groups,
)

logger = logging.getLogger(__name__)


class HAM(Strategy):
    """Hierarchical Adapter Merging (HAM)."""

    def __init__(
        self,
        rank:                 int            = 8,
        num_groups:           int            = 2,
        similarity_threshold: float          = 0.030,
        balance_factor:       float          = 0.4,
        top_k_ratio:          float          = 0.4,
        target_modules:       Optional[List] = None,
        lora_alpha:           float          = 16.0,
        lora_dropout:         float          = 0.01,
        # backwards-compat aliases
        merge_threshold:      Optional[float] = None,
        group_size:           Optional[int]   = None,
    ):
        self.rank               = rank
        self.max_groups         = num_groups
        self.sim_threshold      = merge_threshold if merge_threshold is not None else similarity_threshold
        self.balance_factor     = balance_factor
        self.top_k_ratio        = top_k_ratio
        self.target_modules     = target_modules or ["qkv"]
        self.lora_alpha         = lora_alpha
        self.lora_dropout       = lora_dropout

        self._adapters: List[nn.Module]    = []
        self._alphas:   List[nn.Parameter] = []

        self._groups:       List[Dict[str, Any]] = []
        self._task_to_group: Dict[int, int]      = {}

        self.merged_model: Optional[nn.Module] = None
        self._current_classes: Optional[List[int]] = None

    # ------------------------------------------------------------------

    def on_task_start(self, task_id: int, train_loader, model) -> None:
        model.freeze_backbone()
        model.add_lora(
            task_id,
            rank           = self.rank,
            target_modules = self.target_modules,
            lora_alpha     = self.lora_alpha,
            lora_dropout   = self.lora_dropout,
        )
        self._alphas.append(nn.Parameter(torch.ones(1, device=model._device)))
        self._current_classes = None

    def compute_loss(self, batch, model, task_id: int) -> torch.Tensor:
        x, y = batch[0].to(model._device), batch[1].to(model._device)

        if self._current_classes is None:
            self._current_classes = sorted(y.unique().tolist())

        features = self._forward(x, task_id, model)
        logits   = model.get_head()(features)

        nc       = logits.size(1)
        mask_idx = torch.tensor(self._current_classes, dtype=torch.long, device=model._device)
        not_mask = torch.ones(nc, dtype=torch.bool, device=model._device)
        not_mask[mask_idx] = False
        logits   = logits.index_fill(1, torch.where(not_mask)[0], float("-inf"))

        return F.cross_entropy(logits, y)

    def on_task_end(self, task_id: int, train_loader, model) -> None:
        model.freeze_lora(task_id)
        weights = model.eject_lora(task_id)
        lora_cfg = LoraConfig(
            r=self.rank, lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout, bias="none",
        )
        adapter = build_adapter_from_weights(model, weights, lora_cfg)
        self._adapters.append(adapter)
        self._assign_group(task_id, adapter, self._alphas[task_id].item())

    def extra_params(self) -> List[nn.Parameter]:
        return [self._alphas[-1]] if self._alphas else []

    def evaluate_task(self, test_loader, model, task_id: int):
        device = model._device
        model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y     = x.to(device), y.to(device)
                features = self._forward(x, task_id, model)
                logits   = model.get_head()(features)
                total_loss += F.cross_entropy(logits, y).item()
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)
        model.train()
        return {"accuracy": 100.0 * correct / max(total, 1),
                "loss": total_loss / max(len(test_loader), 1)}

    # ------------------------------------------------------------------

    def _forward(self, x: torch.Tensor, task_id: int, model) -> torch.Tensor:
        """Combine base features with normalised adapter residuals."""
        base = model.encode(x)
        out  = base

        for i in range(min(task_id + 1, len(self._adapters))):
            adapter = self._adapters[i]
            alpha   = self._alphas[i]

            ctx = torch.no_grad() if i < task_id else torch.enable_grad()
            with ctx:
                a_out = (adapter.base_model(x) if hasattr(adapter, "base_model")
                         else adapter(x))

            residual = a_out - base.detach()
            norm     = torch.norm(residual, dim=1, keepdim=True)
            if norm.min() > 0:
                residual = residual / norm
            out = out + alpha.to(x.device) * residual

        return out

    # ------------------------------------------------------------------

    def _assign_group(self, task_id: int, adapter: nn.Module, alpha: float) -> None:
        if task_id == 0 or not self._groups:
            self._groups.append({"tasks": [task_id], "size": 1, "alpha": alpha})
            self._task_to_group[task_id] = 0
            return

        best_idx = self._best_group(adapter)

        if best_idx >= 0:
            g = self._groups[best_idx]
            g["size"] += 1
            g["tasks"].append(task_id)
            g["alpha"] = (g["alpha"] * (g["size"] - 1) + alpha) / g["size"]
            self._task_to_group[task_id] = best_idx
        else:
            idx = len(self._groups)
            self._groups.append({"tasks": [task_id], "size": 1, "alpha": alpha})
            self._task_to_group[task_id] = idx

        logger.info(f"HAM: task {task_id} → group {self._task_to_group[task_id]}")

    def _best_group(self, new_adapter: nn.Module) -> int:
        best_sim, best_idx = -1.0, -1
        total_size = sum(g["size"] for g in self._groups)

        for i, g in enumerate(self._groups):
            sims = [
                self._sim(new_adapter, self._adapters[t])
                for t in g["tasks"] if t < len(self._adapters)
            ]
            avg  = float(np.mean(sims)) if sims else 0.0
            adj  = avg - self.balance_factor * (g["size"] / max(1, total_size))

            if avg > self.sim_threshold and adj > best_sim:
                best_sim, best_idx = adj, i

        if best_idx < 0 and len(self._groups) >= self.max_groups:
            best_idx = int(np.argmax([
                float(np.mean([self._sim(new_adapter, self._adapters[t])
                               for t in g["tasks"] if t < len(self._adapters)]) or 0)
                for g in self._groups
            ]))

        return best_idx

    def _sim(self, a1: nn.Module, a2: nn.Module) -> float:
        return calculate_adapter_similarity(
            a1, a2, method="cosine", layer_filter="blocks.11"
        )

    # ------------------------------------------------------------------

    def perform_final_merge(
        self,
        model,
        combination_type: str   = "dare_ties",
        density:          float = 0.2,
    ) -> None:
        """Two-level merge: concat adapters within each group, then DARE-TIES across groups."""
        max_group_size = max(g["size"] for g in self._groups)

        group_peft_models = []
        for g in self._groups:
            task_adapters = [self._adapters[t] for t in g["tasks"]
                             if t < len(self._adapters)]
            task_alphas   = [self._alphas[t]   for t in g["tasks"]
                             if t < len(self._alphas)]
            combined = concat_lora_adapters(
                task_adapters  = task_adapters,
                task_alphas    = task_alphas,
                rank           = self.rank,
                base_model     = model,
                target_modules = self.target_modules,
                lora_alpha     = 1.0,
                lora_dropout   = self.lora_dropout,
                top_k_ratio    = self.top_k_ratio,
                pad_to_size    = max_group_size,
            )
            group_peft_models.append(combined)

        group_weights = [g["alpha"] for g in self._groups]
        self.merged_model = merge_lora_groups(
            group_adapters   = group_peft_models,
            group_weights    = group_weights,
            base_model       = model,
            combination_type = combination_type,
            density          = density,
        )
        logger.info("HAM: hierarchical merge complete.")

    # ------------------------------------------------------------------

    def get_group_info(self) -> Dict[str, Any]:
        return {
            "num_groups": len(self._groups),
            "groups": [
                {"group_id": i, "size": g["size"],
                 "alpha": g["alpha"], "tasks": g["tasks"]}
                for i, g in enumerate(self._groups)
            ],
        }
