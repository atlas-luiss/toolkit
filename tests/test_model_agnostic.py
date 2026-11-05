"""Any model, any input structure: wrap(), dict (tokenised-text) batches, Hugging Face, LoRA placement,
logging and the CLI."""

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from conftest import MLP, gauss_benchmark, gauss_data
import cl_toolkit as clt
from cl_toolkit import ContinualTrainer, Scenario, adapters
from cl_toolkit import strategies as S
from cl_toolkit.benchmarks import split_by_class


# ------------------------------------------------------------------ dict inputs ("tokenised text")
class TokenData(Dataset):
    """Each class c emits token ids from its own vocabulary slice; x is a dict like a tokenizer output."""

    def __init__(self, n_classes=8, per_class=40, length=12, vocab=200, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items, self.targets = [], []
        for c in range(n_classes):
            lo = c * (vocab // n_classes)
            for _ in range(per_class):
                ids = torch.randint(lo, lo + vocab // n_classes, (length,), generator=g)
                ids[torch.rand(length, generator=g) < 0.3] = torch.randint(0, vocab, (1,), generator=g)
                self.items.append({"input_ids": ids, "attention_mask": torch.ones(length, dtype=torch.long)})
                self.targets.append(c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.targets[i]


class BagEncoder(nn.Module):
    """Mean of token embeddings -> MLP. Accepts dict inputs, like a HF model."""

    def __init__(self, vocab=200, d=48):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.fc1 = nn.Linear(d, d)
        self.hidden_size = d

    def forward(self, input_ids, attention_mask):
        m = attention_mask.unsqueeze(-1).float()
        h = (self.emb(input_ids) * m).sum(1) / m.sum(1)
        return torch.relu(self.fc1(h)) + h


def token_bench():
    return split_by_class(TokenData(seed=0), TokenData(per_class=15, seed=1), num_tasks=4, shuffle_classes=False)


DICT_STRATEGIES = {
    "er": lambda: S.ExperienceReplay(64),
    "derpp": lambda: S.DERPP(64),
    "agem": lambda: S.AGEM(64),
    "icarl": lambda: S.iCaRL(64),
    "gdumb": lambda: S.GDumb(64, epochs=3),
    "lora_er": lambda: S.LoRAReplay(64, rank=4),
    "ewc": lambda: S.EWC(lam=10),
    "lwf": lambda: S.LwF(),
    "fecam": lambda: S.WithHead(S.FirstSessionAdaptation(), clt.heads.FeCAM()),
    "routed": lambda: S.Routed("lora", "lda", experts_kwargs={"rank": 4}),
}


@pytest.mark.parametrize("name", list(DICT_STRATEGIES))
def test_dict_inputs_work_with_every_family(name):
    torch.manual_seed(0)
    model = clt.models.wrap(BagEncoder(), device="cpu")
    assert model.embed_dim == 48
    scen = Scenario(token_bench(), "class", batch_size=16, num_workers=0)
    r = ContinualTrainer(model, scen, DICT_STRATEGIES[name](), lr=5e-3, epochs_per_task=3, verbose=False).fit()
    assert np.isfinite(r["avg_accuracy"]) and r["avg_accuracy"] > 100 / 8, (name, r["avg_accuracy"])


def test_wrap_any_module_tensor_inputs():
    enc = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))
    with pytest.raises(ValueError, match="embed_dim"):
        clt.models.wrap(enc)
    r = clt.run(gauss_data(10), model=clt.models.wrap(enc, embed_dim=64, device="cpu"), strategy="lora-er",
                num_tasks=2, epochs=2, num_workers=0, device="cpu", verbose=False, out_dir=None)
    assert r["avg_accuracy"] > 20


# ------------------------------------------------------------------ Hugging Face (tiny local BERT)
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    d = tmp_path_factory.mktemp("bert")
    words = [f"w{i}" for i in range(60)]
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words))
    transformers.BertTokenizer(str(d / "vocab.txt")).save_pretrained(d)
    cfg = transformers.BertConfig(vocab_size=65, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=64, max_position_embeddings=64)
    torch.manual_seed(0)
    transformers.BertModel(cfg).save_pretrained(d)
    return str(d)


def text_bench(transforms):
    from cl_toolkit.benchmarks import from_texts
    rng = np.random.default_rng(0)
    def make(n):
        texts, labels = [], []
        for c in range(4):
            for _ in range(n):
                texts.append(" ".join(f"w{c * 15 + int(rng.integers(15))}" for _ in range(8)))
                labels.append(c)
        return texts, labels
    return from_texts(make(30), make(10), num_tasks=2, transforms=transforms, shuffle_classes=False)


def test_hf_model_end_to_end_with_lora(tiny_bert):
    m = clt.models.HFModel(tiny_bert, max_length=12, device="cpu")
    assert m.pooling == "cls" and m.embed_dim == 32
    bench = text_bench(m.transforms())
    x, _ = bench.get_splits()[0].train_dataset[0]
    assert set(x) == {"input_ids", "attention_mask"} and x["input_ids"].shape == (12,)
    r = clt.run(bench, model=m, strategy="lora-er", strategy_kwargs={"rank": 4, "memory_size": 40},
                num_tasks=2, epochs=3, lr=5e-3, num_workers=0, device="cpu", verbose=False, out_dir=None)
    assert np.isfinite(r["avg_accuracy"])
    mods = adapters.resolve_lora_targets(m)
    assert mods and all(n.split(".")[-1] in ("query", "value") for n in mods)


# ------------------------------------------------------------------ LoRA placement
def test_lora_targets_and_layers():
    m = MLP()
    assert adapters.lora_candidates(m) == ["0", "qkv"]
    assert adapters.resolve_lora_targets(m, ["qkv"], [2]) == ["2.qkv"]
    assert adapters.resolve_lora_targets(m, r"1\.qkv") == ["1.qkv"]
    with pytest.raises(ValueError, match="Available layer names"):
        adapters.resolve_lora_targets(m, ["q_proj"])
    m.lora_targets, m.lora_layers = ["qkv"], [1]                    # model-level default
    assert adapters.resolve_lora_targets(m) == ["1.qkv"]


def test_lora_strategy_respects_targets():
    torch.manual_seed(0)
    model = MLP()
    scen = Scenario(gauss_benchmark(num_tasks=2), "class", batch_size=32, num_workers=0)
    ContinualTrainer(model, scen, S.LoRAStrategy(rank=4, targets=["qkv"], layers=[2]), epochs_per_task=1,
                     verbose=False).fit()
    lora_modules = {n.rsplit(".lora_", 1)[0] for n, _ in model._backbone.named_parameters() if "lora_" in n}
    assert lora_modules == {"model.2.qkv"}


def test_run_lora_targets_argument():
    tr = clt.build(gauss_data(10), model=MLP(), strategy="lora", lora_targets=["qkv"], lora_layers=[1],
                   num_tasks=2, num_workers=0, device="cpu", verbose=False)
    assert tr.model.lora_targets == ["qkv"] and tr.model.lora_layers == [1]


# ------------------------------------------------------------------ logging + CLI
def test_run_writes_log_config_results(tmp_path):
    out = tmp_path / "run1"
    r = clt.run(gauss_data(10), model=MLP(), strategy="er", num_tasks=2, epochs=1, num_workers=0, device="cpu",
                out_dir=str(out))
    for f in ("log.txt", "config.json", "results.json"):
        assert (out / f).exists(), f
    assert "Accuracy matrix" in (out / "log.txt").read_text()
    res = json.loads((out / "results.json").read_text())
    assert res["avg_accuracy"] == pytest.approx(r["avg_accuracy"], abs=1e-3)
    assert json.loads((out / "config.json").read_text())["strategy"] == "er"


def test_cli_list(capsys):
    from cl_toolkit.__main__ import main
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "strategy" in out and "router" in out and "classifier" in out


def test_available_names_all_resolve():
    av = clt.available()
    for s in av["strategy"]:
        if s in clt.prompts.PROMPTS:
            continue
        clt.strategies.get_strategy(s)
    for h in av["classifier"]:
        if h not in ("linear", "cosine"):
            clt.heads.get_head(h)
    for r in av["router"]:
        if not r.startswith("<"):
            clt.routing.get_router(r)
