"""Jittor dataloading utilities for CODA-Prompt."""

from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloaders.dataloader import iCIFAR10, iCIFAR100, iDOMAIN_NET, iIMAGENET_R
from dataloaders.utils import get_transform


DATASET_REGISTRY = {
	'CIFAR10': (iCIFAR10, 10, [32, 32, 3]),
	'CIFAR100': (iCIFAR100, 100, [32, 32, 3]),
	'ImageNet_R': (iIMAGENET_R, 200, [224, 224, 3]),
	'DomainNet': (iDOMAIN_NET, 345, [224, 224, 3]),
}


def get_dataset_entry(name: str):
	return DATASET_REGISTRY.get(name)


def build_transform(dataset: str, phase: str, aug: bool, resize_imnet: bool):
	return get_transform(dataset=dataset, phase=phase, aug=aug, resize_imnet=resize_imnet)


class LoaderWrapper:
	def __init__(self, dataset, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.drop_last = drop_last
		self.num_workers = num_workers

	def __iter__(self):
		torch_loader = DataLoader(
			self.dataset,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			drop_last=self.drop_last,
			num_workers=self.num_workers,
		)
		for images, targets, task in torch_loader:
			if isinstance(images, torch.Tensor):
				np_images = images.detach().cpu().numpy()
			else:
				np_images = np.asarray(images)
			if isinstance(targets, torch.Tensor):
				np_targets = targets.detach().cpu().numpy()
			else:
				np_targets = np.asarray(targets)
			if isinstance(task, torch.Tensor):
				np_task = task.detach().cpu().numpy()
			else:
				np_task = np.asarray(task)
			yield np_images, np_targets, np_task


def build_loader(dataset, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int):
	return LoaderWrapper(dataset, batch_size, shuffle, drop_last, num_workers)


__all__ = ['get_dataset_entry', 'build_transform', 'build_loader']
