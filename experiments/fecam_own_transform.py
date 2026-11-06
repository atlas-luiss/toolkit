"""FeCAM with the authors' own ViT transform (ToTensor -> Resize(224), bilinear, no normalisation), seeds 0-2."""
import os
import sys
import torchvision.transforms as T
import cl_toolkit as clt
from cl_toolkit.benchmarks.datasets import cifar100

CIFAR_ROOT = os.path.join(os.environ.get("CL_DATA", "./data"), "cifar100")   # set CL_DATA to your data folder
seed = int(sys.argv[1])
tf = T.Compose([T.ToTensor(), T.Resize(224, antialias=True)])
model = clt.models.ViTModel("vit_base_patch16_224.augreg_in21k", freeze_backbone=True)
bench = cifar100(CIFAR_ROOT, 10, transforms=(tf, tf), seed=seed)
clt.run(bench, model=model, strategy="frozen", classifier="fecam", epochs=0, batch_size=128, num_workers=8,
        seed=seed, out_dir=f"runs/repro/fecam_authors_transform_s{seed}")
