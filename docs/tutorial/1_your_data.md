# 1. Your data

**What the toolkit needs:** a training set and a test set, each a normal PyTorch `Dataset` where
`dataset[i]` returns a pair `(x, y)`:

* `x` is the input: an image tensor, a vector, or a dict of tensors for tokenised text
* `y` is the class, an integer `0, 1, 2, ...`

That's all. Splitting the data into tasks comes in the next step. Here are three ways to get there.

## a) Data you already have as tensors (or any PyTorch dataset)

```python
import torch
from torch.utils.data import TensorDataset

x = torch.randn(1200, 32)                 # 1200 samples, 32 features each
y = x[:, :6].argmax(1)                    # 6 classes (a learnable rule, for this example)

train = TensorDataset(x[:1000], y[:1000])
test  = TensorDataset(x[1000:], y[1000:])

print(train[0])                           # (tensor of 32 values, class id)
```

Any dataset that returns `(x, y)` works the same way, for example torchvision's
`CIFAR10(root, train=True, download=True, transform=...)`, or your own class with
`__getitem__` and `__len__`:

```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs, self.labels = inputs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.inputs[i], int(self.labels[i])

my_train = MyDataset(x[:1000], y[:1000])
my_test  = MyDataset(x[1000:], y[1000:])
```

## b) Images in folders

Put one sub-folder per class. The folder names become the class names:

```
my_images/
  cat/    001.jpg  002.jpg ...
  dog/    001.jpg ...
  bird/   ...
```

The toolkit holds out 20% of each class as the test set. If you already have a split, use
`my_images/train/<class>/...` and `my_images/test/<class>/...` instead.

Here we create such a folder with a few generated pictures, then load it:

```python
import os, tempfile
from PIL import Image
from cl_toolkit.benchmarks import from_folder

root = os.path.join(tempfile.mkdtemp(), "my_images")
for c, colour in enumerate(["red", "green", "blue", "yellow"]):
    os.makedirs(f"{root}/class_{c}")
    for i in range(10):
        Image.new("RGB", (64, 64), colour).save(f"{root}/class_{c}/{i}.png")

images = from_folder(root, num_tasks=2)          # already split into 2 tasks (see step 2)
print(images)
print(images.class_name(0))                      # class names come from the folder names
```

With `clt.run` you can simply pass the path: `clt.run(data="/path/to/my_images", ...)`.

## c) Synthetic data (to try things quickly)

The toolkit has two small built-in datasets that need no download:

```python
from cl_toolkit.benchmarks import toy_data

train, test = toy_data()                           # 20 classes of 32-dim vectors
train_img, test_img = toy_data(image_size=224, train_per_class=10)   # 20 classes of 224x224 images
print(len(train), train[0][0].shape, "|", len(train_img), train_img[0][0].shape)
```

They are also available by name: `data="toy"` and `data="toy-images"`. Their accuracies say
nothing about real data; they are for learning the API and testing code.

Named real datasets are downloaded for you: `data="cifar100"`, `"cifar10"`, and, from a folder you
provide, `"imagenet-r"`, `"imagenet-a"`, `"cub200"`. Text: `data="hf:fancyzhx/ag_news"` (any
Hugging Face text-classification dataset).

**Next:** [2. Create the benchmark](2_benchmark.md)
