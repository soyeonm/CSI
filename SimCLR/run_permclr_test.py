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
from sklearn.metrics import roc_auc_score
import numpy as np

import os

#Import default parser from run.py

parser.add_argument('--permclr_views', type=int, default=4)
parser.add_argument('--usual_nll', action='store_true')
parser.add_argument('--text_file_name', type=str, required=True)
#parser.add_argument('--ood', action='store_true')
parser.add_argument('--hydrant_ood', action='store_true')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('-itc', '--intentionally_change_test_classes', action='store_true')
parser.add_argument('--smaller_data', action='store_true')

parser.add_argument('--not_just_average', action='store_true')
parser.add_argument('--train_batch_size', type=int, default=1)
parser.add_argument('-pc', '--p_classifer', action='store_true')
parser.add_argument('-ic', '--indicator_classifier', action='store_true')
parser.add_argument('-s', '--sanity', action='store_true')
parser.add_argument('-d', '--dump', action='store_true')



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
	if args.sanity:
		test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/train'
	else:
		test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/largerco3d/test'
	if not(args.hydrant_ood):
		ood_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj/ood'
	else:
		ood_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj/new_ood_apple_hydrant_toilet'
	if args.smaller_data:
		train_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj/train'
		test_root_dir = '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj/test'

	train_datasets = []
	test_datasets = []
	ood_datasets = []

	classes = [g.split('/')[-1] for g in glob(train_root_dir + '/*')]
	test_classes = set([g.split('/')[-1] for g in glob(test_root_dir + '/*')])
	ood_classes = set([g.split('/')[-1] for g in glob(ood_root_dir + '/*')])

	args.classes_to_idx = {c: i for i, c in enumerate(sorted(classes))}
	for c in classes:
		assert c in test_classes
	assert len(classes) == len(test_classes)
	test_classes = classes
	#Deliberately change test classes
	if args.intentionally_change_test_classes:
		test_classes = classes[1:] + classes[0:1]
	print("test classes are ", test_classes)
	print("classes are ", classes)
	tf = open('test_logs/test_' + args.text_file_name +'.txt', 'w')
	of = open('test_logs/ood_' + args.text_file_name +'.txt', 'w')
	tf.write("classes are " + str(classes) + '\n')
	of.write("classes are " + str(classes) + '\n')
	tf.write("test classes are " + str(test_classes) + '\n')
	of.write("ood classes are " + str(ood_classes) + '\n')
	#tf.close()
	#of.close()

	print("preparing datasets")
	start = time.time()
	for c in classes:
		train_datasets.append(PermDataset(train_root_dir, c, args.permclr_views, args.resize_co3d))
	for c in test_classes:
		test_datasets.append(PermDataset(test_root_dir, c, args.permclr_views, args.resize_co3d))
	
	for c in ood_classes:
		ood_datasets.append(PermDataset(ood_root_dir, c, args.permclr_views, args.resize_co3d))
	if args.dump:
		pickle.dump(test_datasets, open("temp_pickles/permclr_test_datasets.p", "wb"))
		pickle.dump(ood_datasets, open("temp_pickles/permclr_ood_datasets.p", "wb"))
	print("preepared all c! time: ", time.time() - start) 
	print("test set lengths: ", [len(d) for d in test_datasets])
	print("ood set lengths: ", [len(d) for d in ood_datasets])

	test_lens = [len(d) for d in test_datasets]
	#get the max
	argmax = np.argmax(test_lens); max_len = test_lens[argmax]
	#Append 


	test_data_loaders = []
	#dataloaders
	print("preparing dataloaders")
	start = time.time()
	for i, c in enumerate(test_classes):
		test_data_loaders.append(torch.utils.data.DataLoader(test_datasets[i], batch_size=args.batch_size,num_workers=args.workers, pin_memory=True))

	ood_data_loaders = []
	for i, c in enumerate(ood_classes):
		ood_data_loaders.append(torch.utils.data.DataLoader(ood_datasets[i], batch_size=args.batch_size,num_workers=args.workers, pin_memory=True))
	print("preepared all c dataloaders! time: ", time.time() - start)
	if args.dump:
		pickle.dump(test_data_loaders, open("temp_pickles/permclr_test_data_loaders.p", "wb"))
		pickle.dump(ood_data_loaders, open("temp_pickles/permclr_ood_data_loaders.p", "wb"))
	#Model
	model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

	#Load pretrained
	checkpoint = torch.load('saved_models/' + args.model_name, map_location=torch.device('cpu'))
	#state_dict = checkpoint['state_dict']
	model.load_state_dict(checkpoint)

	just_average=True
	get_cutoff = False
	if args.not_just_average:
		just_average=False
		get_cutoff = True


	#Run inference for test
	with torch.cuda.device(args.gpu_index):
		args.ood = False
		permclr = PermCLR(model=model, optimizer=None, scheduler=None, args=args)
		auroc_max_logits_test, auroc_labels_test = permclr.inference(train_datasets, test_datasets, test_data_loaders, tf, just_average, args.train_batch_size, args.p_classifer, get_cutoff) #get cutoff

	#Run inference for ood
	with torch.cuda.device(args.gpu_index):
		args.ood = True
		permclr = PermCLR(model=model, optimizer=None, scheduler=None, args=args)
		auroc_max_logits_ood, auroc_labels_ood = permclr.inference(train_datasets, ood_datasets, ood_data_loaders, of, just_average, args.train_batch_size, args.p_classifer, get_cutoff)

	pickle.dump((auroc_max_logits_test, auroc_labels_test),open("logits/test_" + args.text_file_name + ".p", "wb"))
	pickle.dump((auroc_max_logits_ood, auroc_labels_ood),open("logits/ood_" + args.text_file_name + ".p", "wb"))

	#imbalanced auroc
	imbalanced_auroc = roc_auc_score(np.array(auroc_labels_test+auroc_labels_ood), np.array(auroc_max_logits_test+auroc_max_logits_ood))
	print("Imbalanced auroc is ", imbalanced_auroc)

	#balanced auroc
	if len(auroc_max_logits_ood) < len(auroc_max_logits_test):
		np.random.seed(0)
		balanced_chosen = np.random.permutation(len(auroc_max_logits_ood))
		auroc_max_logits_test = np.array(auroc_max_logits_test)[balanced_chosen].tolist() 
		auroc_labels_test = np.array(auroc_labels_test)[balanced_chosen].tolist()
	else:
		np.random.seed(0)
		balanced_chosen = np.random.permutation(len(auroc_max_logits_test))
		auroc_max_logits_ood = np.array(auroc_max_logits_ood)[balanced_chosen].tolist()
		auroc_labels_ood = np.array(auroc_labels_ood)[balanced_chosen].tolist()
	balanced_auroc = roc_auc_score(np.array(auroc_labels_test+auroc_labels_ood), np.array(auroc_max_logits_test+auroc_max_logits_ood))
	print("Balanced auroc is ", balanced_auroc)

def main_permclr_ood():
	pass

if __name__ == "__main__":
	
	main_permclr_test()



