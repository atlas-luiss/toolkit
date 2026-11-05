"""FeCAM with the authors' own ViT transform (ToTensor -> Resize(224), bilinear, no normalisation), seeds 0-2."""
import sys
import torchvision.transforms as T
import cl_toolkit as clt
from cl_toolkit.benchmarks.datasets import cifar100
seed = int(sys.argv[1])
tf = T.Compose([T.ToTensor(), T.Resize(224, antialias=True)])
model = clt.models.ViTModel("vit_base_patch16_224.augreg_in21k", freeze_backbone=True)
bench = cifar100("/scratch/eric/data/cifar100", 10, transforms=(tf, tf), seed=seed, download=False)
clt.run(bench, model=model, strategy="frozen", classifier="fecam", epochs=0, batch_size=128, num_workers=8,
        seed=seed, out_dir=f"runs/repro/fecam_debug/own_transform_s{seed}")
