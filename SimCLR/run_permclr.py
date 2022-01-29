#run function of permclr

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


#Import default parser from run.py

parser.add_argument('--permclr_views', type=int, default=4)


def main_permclr():
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
	train_datasets = []
	test_datasets = []
	classes = [g.split('/')[-1] for g in glob(train_root_dir + '/*')]
	test_classes = set([g.split('/')[-1] for g in glob(test_root_dir + '/*')])
	args.classes_to_idx = {c: i for i, c in enumerate(sorted(classes))}
	for c in classes:
		assert c in test_classes
	assert len(classes) == len(test_classes)
	print("classes are ", classes)

	print("preparing datasets")
	start = time.time()
	for c in classes:
		train_datasets.append(PermDataset(train_root_dir, c, args.permclr_views, args.resize_co3d))
		print("done training for ", c)
		test_datasets.append(PermDataset(test_root_dir, c, args.permclr_views, args.resize_co3d))
		print("done test for ", c)
	print("preepared all c! time: ", time.time() - start)
	pickle.dump(train_datasets[0][0], open("original.p", "wb"))


	train_data_loaders = []
	test_data_loaders = []
	#dataloaders
	print("preparing dataloaders")
	start = time.time()
	for i, c in enumerate(classes):
		train_data_loaders.append(torch.utils.data.DataLoader(train_datasets[i], batch_size=args.batch_size,num_workers=args.workers, pin_memory=True))
		test_data_loaders.append(torch.utils.data.DataLoader(test_datasets[i], batch_size=args.batch_size,num_workers=args.workers, pin_memory=True))
	print("preepared all c dataloaders! time: ", time.time() - start)

	#Model
	model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
	optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

	if args.load_pretrained:
		checkpoint = torch.load('../simclr_embeddings/CIFAR10_resnet18/checkpoint_0100.pth.tar', map_location=torch.device('cpu'))
		state_dict = checkpoint['state_dict']
		model.load_state_dict(state_dict)

	#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
	#													   last_epoch=-1)
	#Training epoch
	#with torch.cuda.device(args.gpu_index):
	#    simclr = PermCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
	#    #TODO: implement PermCLR
	#    simclr.train(train_loader)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=min([len(train_loader) for train_loader in train_data_loaders]), eta_min=0,
														   last_epoch=-1)
	#print("shuffling dataloaders")
	#for epoch in range(10):
	#	for i, c in enumerate(classes):
	#		train_datasets[i].shuffle()
	#		train_data_loaders[i] = torch.utils.data.DataLoader(train_datasets[i], batch_size=args.batch_size,num_workers=args.workers, pin_memory=True)
	#	pickle.dump(train_datasets[0][0], open("shuffled.p", "wb"))
	#print("shuffled all c dataloaders! time: ", time.time() - start)
	with torch.cuda.device(args.gpu_index):
		permclr = PermCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
		permclr.train(train_datasets, train_data_loaders)




if __name__ == "__main__":
	
	main_permclr()