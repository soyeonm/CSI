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


parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--sanity', action='store_true')
parser.add_argument('--inf_workers', type=int, default=1)
parser.add_argument('--eval_train_batch_size', type=int, default=10)
parser.add_argument('--eval_test_batch_size', type=int, default=16)

args = parser.parse_args()


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
train_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_9_classify/train'
test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_9_classify_real/test'
sample=None
if args.sanity:
	test_root_dir = train_root_dir
	sample = 0.1

state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

permclr_train_dataset = ObjInferenceDataset(train_root_dir, args.object_views, resize_shape= args.resize_co3d, shots=args.eval_train_batch_size,  transform=None, processed=processed)
train_class_idx = permclr_train_dataset.class2idx

test_dataset = ObjInferenceDataset(test_root_dir, args.object_views, resize_shape= args.resize_co3d, shots=None,  transform=None, processed=True, class_idx=train_class_idx, sample=sample)

test_data_loader = MultiEpochsDataLoader(test_dataset, batch_size=args.eval_test_batch_size, num_workers=args.inf_workers, pin_memory=False, shuffle=False, persistent_workers=True)

args.ood = False
objclr = ObjCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
objclr.classify_inference(permclr_train_dataset, test_data_loader, f, just_average=True, train_batch_size=args.eval_train_batch_size)







