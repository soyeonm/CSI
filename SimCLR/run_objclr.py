from supcontrast_util import TwoCropTransform

from run import *
import torch
from data_aug.permclr_custom_dataset import PermDataset
from data_aug.objclr_custom_dataset import ObjDataset
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

parser.add_argument('--object_views', type=int, default=4)
#parser.add_argument('--usual_nll', action='store_true')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--smaller_data', action='store_true')
#parser.add_argument('--debug_with_identity', action='store_true')
#parser.add_argument('--num_perms', type=int, default=4)
parser.add_argument('--class_label', action='store_true')


def main_objclr():
	args = parser.parse_args()
	if not args.disable_cuda and torch.cuda.is_available():
		args.device = torch.device('cuda')
		cudnn.deterministic = True
		cudnn.benchmark = True
	else:
		args.device = torch.device('cpu')
		args.gpu_index = -1

	train_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/train'
	#Add transform later
	train_dataset = ObjDataset(train_root_dir, args.object_views,  transform=ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(args.co3d_cropsize, 1, args.resize_co3d), 2)) #transform can be None too
	pickle.dump(train_dataset, open("objclr_train_dataset.p", "wb"))
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True, drop_last=True)

	model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
	if args.load_pretrained:
		checkpoint = torch.load('../simclr_embeddings/CIFAR10_resnet18/checkpoint_0100.pth.tar', map_location=torch.device('cpu'))
		state_dict = checkpoint['state_dict']
		model.load_state_dict(state_dict)

	optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
														   last_epoch=-1)

	# Permclr train datasets/ test datasets for inference
	#SANITY
	test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/train'
	permclr_train_datasets = []
	test_datasets = []

	classes = [g.split('/')[-1] for g in glob(train_root_dir + '/*')]
	test_classes = set([g.split('/')[-1] for g in glob(test_root_dir + '/*')])
	args.classes_to_idx = {c: i for i, c in enumerate(sorted(classes))}
	print("test classes are ", test_classes)
	print("classes are ", classes)

	print("preparing datasets")
	start = time.time()
	for c in classes:
		permclr_train_datasets.append(PermDataset(train_root_dir, c, args.object_views, args.resize_co3d))
	for c in test_classes:
		test_datasets.append(PermDataset(test_root_dir, c, args.object_views, args.resize_co3d))
	print("preepared all c! time: ", time.time() - start) 

	test_data_loaders = []
	print("preparing dataloaders")
	start = time.time()
	for i, c in enumerate(test_classes):
		test_data_loaders.append(torch.utils.data.DataLoader(test_datasets[i], batch_size=1,num_workers=args.workers, pin_memory=True))
	print("preepared all c dataloaders! time: ", time.time() - start)




	#  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
	with torch.cuda.device(args.gpu_index):
		args.ood = False
		start = time.time()
		objclr = ObjCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
		objclr.train(train_loader, permclr_train_datasets, test_data_loaders, just_average=True, train_batch_size=10, class_lens = 3, eval_period = 1)
		print("time taken per epoch is ", time.time() - start)


if __name__ == "__main__":
	main_objclr()