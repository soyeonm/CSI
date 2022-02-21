#helper functions

import pickle
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--num_classes', type=int, default=3)
args = parser.parse_args()

save_total_minus_list = pickle.load(open('logits/cutoff_test.p', 'rb'))

cor_entries_list = [i*args.num_classes + i for i in range(args.num_classes)] #0,4,8

cor_entries = set(cor_entries_list) #0,4,8

#for i, save_total_minus in enumerate(save_total_minus_list):#
#	if 

cor_entry2other_incor_entries = []

#Let's look at the overlap first
cor_dict = {i:[] for i in cor_entries}
incor_dict = {i:[] for i in cor_entries}

last_cor_entry = None
for save_total_minus in save_total_minus_list:
	for i in range(save_total_minus.shape[0]): 
		if i in cor_entries:
			cor_dict[i].append(save_total_minus[i]/ torch.std(save_total_minus[i]))
			#cor_dict[i].append(save_total_minus[i])
			#last_cor_entry = i
		else:
			cor_entry = int(i/args.num_classes)
			incor_dict[cor_entries_list[cor_entry]].append(save_total_minus[i]/ torch.std(save_total_minus[i]))
			#incor_dict[cor_entries_list[cor_entry]].append(save_total_minus[i])

#Sort 

cor_dict_sorted = {i:[] for i in cor_entries}
incor_dict_sorted = {i:[] for i in cor_entries}

#First sort across everything
for i, v in cor_dict.items():
	catted_all = torch.cat(v, axis=0).numpy()
	cor_dict_sorted[i] = np.sort(catted_all)


#Second sort across just each instance
for i, v in incor_dict.items():
	catted_all = torch.cat(v, axis=0).numpy()
	incor_dict_sorted[i] = np.sort(catted_all)


ood_save_total_minus_list = pickle.load(open('logits/cutoff_ood.p', 'rb'))
#Do the same

ood_dict = {i:[] for i in cor_entries}

for save_total_minus in ood_save_total_minus_list:
	for i in range(save_total_minus.shape[0]): 
		cor_entry = int(i/args.num_classes)
		#incor_dict[cor_entries_list[cor_entry]].append(save_total_minus[i]/ torch.std(save_total_minus[i]))
		ood_dict[cor_entries_list[cor_entry]].append(save_total_minus[i]/ torch.std(save_total_minus[i]))

ood_dict_sorted = {i:[] for i in cor_entries}

for i, v in ood_dict.items():
	catted_all = torch.cat(v, axis=0).numpy()
	ood_dict_sorted[i] = np.sort(catted_all)


#ver 1 only get 90% from cor entries
#for entry in cor_entries:

#ver 2 get tpr, fpr idk 