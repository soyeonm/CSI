#Sanity

from supcontrast_util import TwoCropTransform

from run import *
import torch
from data_aug.permclr_custom_dataset import PermDataset
from data_aug.objclr_custom_dataset import ObjDataset, ObjInferenceDataset
from data_aug.view_generator import ContrastiveLearningViewGenerator
from data_aug.contrastive_learning_dataset import get_simclr_pipeline_transform
from glob import glob
import pickle
import time
from objclr import ObjCLR

import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from models.resnet_simclr import ResNetSimCLR

import os

import torch.distributed as dist
import datetime

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

permclr_train_dataset = ObjInferenceDataset(train_root_dir, args.object_views, resize_shape= args.resize_co3d, shots=args.eval_train_batch_size,  transform=None, processed=processed)
train_class_idx = permclr_train_dataset.class2idx

test_dataset = ObjInferenceDataset(test_root_dir, args.object_views, resize_shape= args.resize_co3d, shots=None,  transform=None, processed=True, class_idx=train_class_idx)

test_data_loader = MultiEpochsDataLoader(test_dataset, batch_size=args.eval_test_batch_size, num_workers=args.inf_workers, pin_memory=False, shuffle=False, persistent_workers=True)

args.ood = False
objclr = ObjCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
objclr.classify_inference(permclr_train_dataset, test_data_loader, f, just_average=True, train_batch_size=args.eval_train_batch_size)







