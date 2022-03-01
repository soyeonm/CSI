#helper functions

import pickle
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--num_classes', type=int, default=3)
args = parser.parse_args()

#save_total_minus_list = pickle.load(open('logits/cutoff_test.p', 'rb'))
save_total_minus_list = pickle.load(open('logits/cutoff_train_sanity.p', 'rb'))

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
			#cor_dict[i].append(save_total_minus[i]/ torch.std(save_total_minus[i]))
			cor_dict[i].append(save_total_minus[i])
			#last_cor_entry = i
		else:
			cor_entry = int(i/args.num_classes)
			#incor_dict[cor_entries_list[cor_entry]].append(save_total_minus[i]/ torch.std(save_total_minus[i]))
			incor_dict[cor_entries_list[cor_entry]].append(save_total_minus[i])

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


#####
#Get commands and results
#
#cor dict is 'logits/cutoff_train_sanity.p' here (tmux a -t 0)
>>> s = [torch.sum(torch.abs(cor) < 2e-4) for cor in cor_dict[8]]                                                                                                                                                                                    
>>> sum([ss<32 for ss in s])                                                                                                                                                                                                                         
tensor(16)
>>> s = [torch.sum(torch.abs(cor) < 1.5e-4) for cor in cor_dict[8]]                                                                                                                                                                                  
>>> sum([ss<32 for ss in s])                                                                                                                                                                                                                         
tensor(28)

>>> s = [torch.sum(torch.abs(cor) < 1e-4) for cor in cor_dict[0]]   
>>> s = [torch.sum(torch.abs(cor) < 1e-4) for cor in cor_dict[0]]                                                                                                                                                                                    
>>> sum([ss<32 for ss in s])   
tensor(4)

#sum([ss<32 for ss in s]) 이 p-value를 만족하는 애들의 수 (null reject되서 ood라고 predict될 애들의 수). 이게 적으면 좋은거. 36/360 이랑 32/323 이랑 맞추면 그게 threshold. 
#cor_dict는 이 sum/ len(s)가 작고 ood_dict 는 이게 커야함

#Results
#0 (first class)
#CORDICT sanity
>>> s = [torch.sum(torch.abs(cor) < 3e-5) for cor in cor_dict[0]] 
>>> sum([ss<36 for ss in s])  
tensor(36) #should be 32 actually whatever

#CORDICT test 
>>> s = [torch.sum(torch.abs(cor) < 3e-5) for cor in ood_dict[0]] 
>>> sum([ss<36 for ss in s])
tensor(22)
>>> len(s)
27 #22/27 detection. good performance 

>>> s = [torch.sum(torch.abs(cor) < 3e-5) for cor in cor_dict[0]]    
>>> sum([ss<36 for ss in s])
tensor(6)
>>> 6/82
0.07317073170731707 #calibration good 


#4 (middle class)
#CORDICT sanity
>>> s = [torch.sum(torch.abs(cor) < 1.9e-7) for cor in cor_dict[4]] 
>>> sum([ss<36 for ss in s])  
tensor(23)

#CORDICT test 
>>> s = [torch.sum(torch.abs(cor) < 1.9e-7) for cor in ood_dict[4]]                                                                                                                                                                                  
>>> sum([ss<36 for ss in s])
tensor(27)
>>> len(s)
27 #100% detection. good performance

>>> s = [torch.sum(torch.abs(cor) < 2e-7) for cor in cor_dict[4]]                                                                                                                                                                                    
>>> sum([ss<36 for ss in s])                                                                                                                                                                                                                         
tensor(38)
>>> len(s)
82 #calibration 




#8 (last class)
#CORDICT sanity

>>> s = [torch.sum(torch.abs(cor) < 1.5e-4) for cor in cor_dict[8]]                                                                                                                                                                                  
>>> sum([ss<36 for ss in s])                                                                                                                                                                                                                         
tensor(35)

#CORDICT test 

>>> s = [torch.sum(torch.abs(cor) < 1.5e-4) for cor in ood_dict[8]] 
>>> sum([ss<36 for ss in s])
tensor(14)
>>> 14/27
0.5185185185185185 #performance bad 
>>> s = [torch.sum(torch.abs(cor) < 1.5e-4) for cor in cor_dict[8]] 
>>> sum([ss<36 for ss in s])
tensor(9)
>>> len(s)
82
>>> 9/82
0.10975609756097561 #p value transfer good 

>>> s = [torch.sum(torch.abs(cor) < 1e-5) for cor in cor_dict[8]]                                                                                                                                                                                    
>>> sum([ss<36 for ss in s])                                                                                                                                                                                                                         
tensor(11)
>>> sum([ss<40 for ss in s])
tensor(11)
>>> 40/360
0.1111111111111111
>>> 11/82
0.13414634146341464
>>> sum([ss<360*0.13 for ss in s])
tensor(12)
>>> s = [torch.sum(torch.abs(cor) < 1e-5) for cor in ood_dict[8]] 
>>> sum([ss<360*0.13 for ss in s])
tensor(24)
#-> 바로 잘되



######
### Get AUROC and fpr/ tpr
######

#Search over the correct ranking in cor_dict for 1/82, ...., 81/82
#
#Get all the cor in cor_dict[8]
catted = torch.abs(torch.cat(cor_dict[4]))
#sort 
catted = np.sort(catted.numpy())

#Now go over all of catted to find 1/82, ...
#start from all 82
#These are wrong
thresholds = {i: None for i in range(1, 82)}
current_pointer = 81 #360 * 81/82
c_counter = 0
for c in catted:
	c_counter +=1
	#if c_counter %1000==0:
	#	print("c_counter is ", c_counter)
	s = [torch.sum(torch.abs(cor) <= c) for cor in cor_dict[4]] 
	if sum([ss<360 * current_pointer/82 for ss in s]) <= current_pointer:
		thresholds[current_pointer] = c
		#간단한 버전
		current_pointer += -1

fpr = {}
for i, th in thresholds.items():
	s = [torch.sum(torch.abs(cor) <= th) for cor in ood_dict[4]] 
	fpr[i] = sum([ss<360 * i/82 for ss in s]).item()/len(s)
	  

#Questionable
thresholds_better = {i: None for i in range(1, 82)}
current_pointer = 81 #360 * 81/82
catted = catted.numpy().tolist()
for c_idx, c in enunmerate(catted):
	if c_idx>=1:
		prev_c = catted[c_idx-1]
		prev_s = s
	else:
		prev_c = 0.0
		prev_s = [0.0] * len(cor_dict[4])
	s = [torch.sum(torch.abs(cor) <= c) for cor in cor_dict[4]] 
	if c_idx>=1:
		if sum([ss<360 * current_pointer/82 for ss in prev_s]) <= current_pointer and sum([ss<360 * current_pointer/82 for ss in s]) > current_pointer:
			thresholds[current_pointer] = prev_c
			#간단한 버전
			current_pointer += -1
	else:
		if sum([ss<360 * current_pointer/82 for ss in s]) <= current_pointer:
			thresholds[current_pointer] = c
			#간단한 버전
			current_pointer += -1

#5% threshold
threshold = {10: [3e-5, 2e-6,1.5e-4], 25:[4e-6,1.5e-6,6.3e-6], 50: [3.3e-6, 1e-6, 1.2e-5], 75:[3e-6,8e-7,9e-6] , 100:[1e-6, 1e-7, 1e-7]}


#Let's get ood
ood_right = {}
for th, ths in threshold.items():
	 #Get sum across the three thresholds
	 ood_right_sum = 0
	 for o0, o4, o8 in zip(ood_dict[0], ood_dict[4],ood_dict[8]):
	 	sum0 = torch.sum(torch.abs(o0) <= ths[0])
	 	sum4 = torch.sum(torch.abs(o4) <= ths[1])
	 	sum8 = torch.sum(torch.abs(o8) <= ths[2])
	 	if sum0 <= th/100*360 and sum4 <= th/100*360 and sum8 <= th/100*360: #then ood
	 		ood_right_sum +=1
	 ood_right[th] = ood_right_sum/len(ood_dict[0])



