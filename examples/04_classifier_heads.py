"""Classifier heads from recent papers, on top of any strategy.

    python examples/04_classifier_heads.py
"""
import os

import cl_toolkit as clt

ROOT = os.environ.get("CL_DATA", "./data")
common = dict(data="cifar100", root=f"{ROOT}/cifar100", num_tasks=10, batch_size=128)

# Frozen pre-trained model + a statistical head (no training at all):
for head in ["ncm", "lda", "klda", "fecam", "ranpac"]:
    clt.run(strategy="frozen", classifier=head, epochs=0, **common)

# Correct a trained linear head after each task (SLCA classifier alignment):
clt.run(strategy="naive", classifier="slca", epochs=1, **common)

# Heads compose with any strategy, e.g. replay + FeCAM, or a configured head:
clt.run(strategy="er", strategy_kwargs={"memory_size": 2000}, classifier="fecam", **common)
clt.run(strategy="frozen", classifier="ranpac", classifier_kwargs={"proj_dim": 5000}, epochs=0, **common)

# First-session adaptation (APER / RanPAC): adapt on task 0 only, then a head:
clt.run(strategy="first-session", strategy_kwargs={"first_session": "adaptformer"}, classifier="ranpac",
        epochs=3, **common)
