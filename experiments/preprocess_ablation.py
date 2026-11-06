"""NCM on frozen ViT-B/16 IN21K, CIFAR-100 10 tasks, seed 0: effect of the test-time preprocessing.
Logs: runs/repro/preproc/<variant>/"""
import os
import sys
import torchvision.transforms as T
import cl_toolkit as clt
from cl_toolkit.benchmarks.datasets import cifar100

CIFAR_ROOT = os.path.join(os.environ.get("CL_DATA", "./data"), "cifar100")   # set CL_DATA to your data folder

v = sys.argv[1]
norm = T.Normalize((0.5,) * 3, (0.5,) * 3)
tf = {
    "timm-crop":    None,   # timm data config: Resize(248) + CenterCrop(224) + Normalize(.5,.5)
    "resize-norm":  T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), norm]),
    "resize-nonorm": T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]),
}[v]
model = clt.models.ViTModel("vit_base_patch16_224.augreg_in21k", freeze_backbone=True)
tfs = model.transforms() if tf is None else (tf, tf)
bench = cifar100(CIFAR_ROOT, 10, transforms=tfs, seed=0)
clt.run(bench, model=model, strategy="simplecil", epochs=0, batch_size=128, num_workers=8, seed=0,
        out_dir=f"runs/repro/preproc/{v}")
