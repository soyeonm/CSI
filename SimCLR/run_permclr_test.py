#Run permclr on test set, ood set

#Test set

from run import *
import torch
from data_aug.permclr_custom_dataset import PermDataset
from glob import glob
import pickle
import time
from permclr import PermCLR

import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from models.resnet_simclr import ResNetSimCLR

import os

#Import default parser from run.py

parser.add_argument('--permclr_views', type=int, default=4)
parser.add_argument('--usual_nll', action='store_true')
parser.add_argument('--text_file_name', type=str, required=True)
parser.add_argument('--ood', action='store_true')


def main_permclr_test():
	args = parser.parse_args()
	if not args.disable_cuda and torch.cuda.is_available():
		args.device = torch.device('cuda')
		cudnn.deterministic = True
		cudnn.benchmark = True
	else:
		args.device = torch.device('cpu')
		args.gpu_index = -1

	#Define the dataset
	train_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/train'
	test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/test'
	if args.ood:
		test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one/ood'
	train_datasets = []
	test_datasets = []
	classes = [g.split('/')[-1] for g in glob(train_root_dir + '/*')]
	test_classes = set([g.split('/')[-1] for g in glob(test_root_dir + '/*')])
	args.classes_to_idx = {c: i for i, c in enumerate(sorted(classes))}
	for c in classes:
		assert c in test_classes
	assert len(classes) == len(test_classes)
	print("classes are ", classes)
	f = open('test_logs/' + args.text_file_name +'.txt', 'w')
	f.write("classes are " + str(classes) + '\n')
	f.close()

	print("preparing datasets")
	start = time.time()
	for c in classes:
		train_datasets.append(PermDataset(train_root_dir, c, args.permclr_views, args.resize_co3d))
		test_datasets.append(PermDataset(test_root_dir, c, args.permclr_views, args.resize_co3d))
		print("done test for ", c)
	print("preepared all c! time: ", time.time() - start)

	test_data_loaders = []
	#dataloaders
	print("preparing dataloaders")
	start = time.time()
	for i, c in enumerate(classes):
		test_data_loaders.append(torch.utils.data.DataLoader(test_datasets[i], batch_size=args.batch_size,num_workers=args.workers, pin_memory=True))
	print("preepared all c dataloaders! time: ", time.time() - start)

	#Model
	model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
	optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

	#Load pretrained
	checkpoint = torch.load('saved_models/first_permclr_broccoli_epoch_199', map_location=torch.device('cpu'))
	#state_dict = checkpoint['state_dict']
	model.load_state_dict(checkpoint)

	#Run inference once
	with torch.cuda.device(args.gpu_index):
		permclr = PermCLR(model=model, optimizer=None, scheduler=None, args=args)
		permclr.inference(train_datasets, test_datasets, test_data_loaders)



def main_permclr_ood():
	pass

if __name__ == "__main__":
	
	main_permclr_test()



