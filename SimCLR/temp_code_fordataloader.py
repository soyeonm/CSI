test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/test'
from run import *
#
import torch
from data_aug.permclr_custom_dataset import PermDataset
from glob import glob
import pickle
import time
test_classes = set([g.split('/')[-1] for g in glob(test_root_dir + '/*')])
train_datasets = []
test_datasets = []
#
train_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/train'
classes = [g.split('/')[-1] for g in glob(train_root_dir + '/*')]
start = time.time()
#
for c in classes:
    train_datasets.append(PermDataset(train_root_dir, c, 4, 32))
    print("done training for ", c)
    test_datasets.append(PermDataset(test_root_dir, c, 4, 32))
#
#
train_data_loaders = []
test_data_loaders = []
for i, c in enumerate(classes):
    train_data_loaders.append(torch.utils.data.DataLoader(train_datasets[i], batch_size=2,num_workers=1, pin_memory=True))
    test_data_loaders.append(torch.utils.data.DataLoader(test_datasets[i], batch_size=2,num_workers=1, pin_memory=True))
#
for batch_dict_tuple in zip(*train_data_loaders):
    break
#
for batch_dict in train_data_loaders[0]:
    break