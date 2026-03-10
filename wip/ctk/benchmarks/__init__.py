from .base import CLBenchmark, TaskSplit
from .split import make_class_splits
from .cifar100 import SplitCIFAR100
from .cub200 import SplitCUB200
from .imagenet_r import ImageNetR
from .imagenet_a import ImageNetA
from .stream import CustomStream

__all__ = [
    'CLBenchmark',
    'TaskSplit',
    'make_class_splits',
    'SplitCIFAR100',
    'SplitCUB200',
    'ImageNetR',
    'ImageNetA',
    'CustomStream',
]
