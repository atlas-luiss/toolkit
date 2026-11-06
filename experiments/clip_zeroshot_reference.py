"""Independent CLIP zero-shot on CIFAR-100 test, plain open_clip (no toolkit code)."""
import os
import torch, open_clip, torch.nn.functional as F
from torchvision.datasets import CIFAR100
from cl_toolkit.models.clip import CIFAR_TEMPLATES

CIFAR_ROOT = os.path.join(os.environ.get("CL_DATA", "./data"), "cifar100")   # set CL_DATA to your data folder
model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-B-16")
model = model.cuda().eval()
ds = CIFAR100(CIFAR_ROOT, train=False, download=True, transform=pre)
names = [c.replace("_", " ") for c in ds.classes]
with torch.no_grad():
    W = torch.stack([F.normalize(F.normalize(model.encode_text(tok([t.format(c) for t in CIFAR_TEMPLATES]).cuda()), dim=-1).mean(0), dim=0) for c in names])
    correct = 0
    for x, y in torch.utils.data.DataLoader(ds, batch_size=256, num_workers=8):
        f = F.normalize(model.encode_image(x.cuda()), dim=-1)
        correct += ((f @ W.T).argmax(1).cpu() == y).sum().item()
print("zero-shot top-1:", 100 * correct / len(ds))
print("first names:", names[:5])
