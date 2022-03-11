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

import torch.distributed as dist
import datetime

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') 

parser.add_argument('--object_views', type=int, default=4)
#parser.add_argument('--usual_nll', action='store_true')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--smaller_data', action='store_true')
#parser.add_argument('--debug_with_identity', action='store_true')
#parser.add_argument('--num_perms', type=int, default=4)
parser.add_argument('--class_label', action='store_true')
parser.add_argument('--same_labels_mask', action='store_true')
parser.add_argument('--eval_train_batch_size', type=int, default=10)
parser.add_argument('--sanity', action='store_true')
parser.add_argument("--local_rank", type=int,
						default=0, help='Local rank for distributed learning')
parser.add_argument('--permclr_workers', type=int, default=0)
parser.add_argument('-ood','--ood_classes', nargs='+', help='<Required> Set flag', deafult=[])



############Set torch device for MiltiGPU###
args = parser.parse_args()


if not args.disable_cuda and torch.cuda.is_available():
	torch.cuda.set_device(args.local_rank)
	args.device = torch.device('cuda')
	cudnn.deterministic = True
	cudnn.benchmark = True
else:
	args.device = torch.device('cpu')
	args.gpu_index = -1 #What to do about this?

#device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.n_gpus = torch.cuda.device_count() #Use CUDA_VISIBLE_DEVICES

if args.n_gpus > 1:
	import apex
	import torch.distributed as dist
	from torch.utils.data.distributed import DistributedSampler

	args.multi_gpu = True
	torch.distributed.init_process_group(
		'nccl',
		init_method='env://',
		world_size=args.n_gpus,
		rank=args.local_rank,
	)
	print("local rank is ", args.local_rank)
	print("args.device is ", args.device)
else:
	args.multi_gpu = False


def main_objclr():

	#train_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/train'
	#train_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_9_classify_real/test'
	train_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_9_classify/train'
	#Add transform later
	start = time.time()
	train_dataset = ObjDataset(train_root_dir, args.object_views,  transform=ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(args.co3d_cropsize, 1, args.resize_co3d), 2), processed=False) #transform can be None too
	#pickle.dump(train_dataset, open("objclr_train_dataset.p", "wb"))
	print("loaded train dataset in ", (time.time()- start)/60, " mins!")

	if args.multi_gpu:
		train_sampler = DistributedSampler(train_dataset, num_replicas=args.n_gpus, rank=args.local_rank)
		train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True, drop_last=True) #sampler option is mutually exlusive with shuffle
	else:
		train_sampler = None
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
	model = model.to(device=torch.device('cuda'))
	############MultiGPU
	if args.multi_gpu:
		model = apex.parallel.convert_syncbn_model(model)
		model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
	

	# Permclr train datasets/ test datasets for inference
	#Do the permclr evaluation only on the 0th gpu
	#SANITY
	if args.local_rank ==0:
		if args.sanity:
			test_root_dir = train_root_dir
		else:
			#test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/test'
			test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_9_classify_real/test'
		if args.smaller_data:
			train_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj/train'
			if args.sanity:
				test_root_dir = train_root_dir
			else:
				test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj/test'
		permclr_train_datasets = []
		test_datasets = []

		ood_classes = args.ood_classes
		classes = [g.split('/')[-1] for g in glob(train_root_dir + '/*')]
		classes = [c for c in classes if not(c in set(ood_classes))]
		#test_classes = set([g.split('/')[-1] for g in glob(test_root_dir + '/*')])
		test_classes = classes
		args.classes_to_idx = {c: i for i, c in enumerate(sorted(classes))}
		print("test classes are ", test_classes)
		print("classes are ", classes)
		print("ood classes are ", ood_classes)

		print("preparing datasets")
		start = time.time()
		for c in classes:
			permclr_train_datasets.append(PermDataset(train_root_dir, c, args.object_views, args.resize_co3d, processed=False))
		for c in test_classes:
			test_datasets.append(PermDataset(test_root_dir, c, args.object_views, args.resize_co3d))
		print("preepared all c! time: ", time.time() - start) 

		test_data_loaders = []
		print("preparing dataloaders")
		start = time.time()
		for i, c in enumerate(test_classes):
			test_data_loaders.append(torch.utils.data.DataLoader(test_datasets[i], batch_size=1,num_workers=args.permclr_workers, pin_memory=True))
		print("preepared all c dataloaders! time: ", time.time() - start)
		class_lens = len(classes)
	else:
		test_data_loaders = None
		permclr_train_datasets = None
		class_lens = None

	if args.multi_gpu:
		#dist.monitored_barrier(timeout=datetime.timedelta(0, 30), wait_all_ranks=True)
		dist.barrier()
	


	#  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
	#with torch.cuda.device(args.gpu_index):
	args.ood = False
	start = time.time()
	objclr = ObjCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
	#TODO for args.0th gpu
	objclr.train(train_loader, permclr_train_datasets, test_data_loaders, just_average=True, train_batch_size=args.eval_train_batch_size, class_lens = class_lens, eval_period = 5, train_sampler=train_sampler)
	print("time taken per epoch is ", time.time() - start)

	save_checkpoint(args.epochs, model, args.model_name, 'obj_saved_models', multi_gpu = args.multi_gpu)

if __name__ == "__main__":
	main_objclr()