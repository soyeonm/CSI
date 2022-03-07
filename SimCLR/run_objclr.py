from supcontrast_util import TwoCropTransform

from run import *
import torch
from data_aug.objclr_custom_dataset import ObjDataset
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
	train_dataset = ObjDataset(train_root_dir, args.object_views, args.resize_co3d, transform=ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(32, 1, 32), 2)) #transform can be None too
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

	#  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
	with torch.cuda.device(args.gpu_index):
		start = time.time()
		objclr = ObjCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
		objclr.train(train_loader)
		print("time taken per epoch is ", time.time() - start)


if __name__ == "__main__":
	main_objclr()