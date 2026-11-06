"""Benchmarks: any dataset -> ordered list of tasks."""
from .base import CLBenchmark, TaskSplit
from .split import split_by_class, split_by_domain, few_shot, joint, get_targets, make_class_splits
from .text import TextDataset, from_texts, hf_text
from .toy import toy, toy_data, toy_images
from .datasets import (get_benchmark, from_folder, cifar100, cifar10, default_transforms, DATASETS,
                       SplitCIFAR100, ImageNetR, ImageNetA, SplitCUB200)

__all__ = ["toy", "toy_data", "toy_images", "TextDataset", "from_texts", "hf_text", "CLBenchmark", "TaskSplit", "split_by_class", "split_by_domain", "few_shot", "joint",
           "get_targets", "make_class_splits", "get_benchmark", "from_folder", "cifar100", "cifar10",
           "default_transforms", "DATASETS", "SplitCIFAR100", "ImageNetR", "ImageNetA", "SplitCUB200"]
