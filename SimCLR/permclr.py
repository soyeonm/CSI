import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

import pickle
import time
import numpy as np
import copy

import itertools

torch.manual_seed(0)

#Multiply this matrix to make the feature matrix into A
def get_A_matrix(num_perspectives, num_batch=2):
	#Assume that num_batch is 2
	mat = torch.zeros(num_perspectives*2, num_perspectives*2)
	for i in range(num_perspectives):
		mat[i, 2*i+0] = 1
		mat[num_perspectives+i, 2*i+1] = 1
	return mat

#TODO: code this 
def get_perm_matrix(num_perspectives):
	mat = torch.zeros(2*num_perspectives, 2*num_perspectives) 
	for i in range(2*num_perspectives):
		if i< num_perspectives:
			mat[2*i, i] = 1
		else:
			mat[2*(i-num_perspectives)+1, i]=1
	return mat

def get_perm_matrix_identity(num_perspectives):
	mat = torch.eye(2*num_perspectives, 2*num_perspectives) 
	return mat

def get_perm_matrix_one(num_perspectives, swap_1st, swap_2nd):
	mat = torch.eye(2*num_perspectives, 2*num_perspectives) 
	new_mat = mat.clone()
	new_mat[swap_1st, :] = mat[swap_2nd, :]; new_mat[swap_2nd, :] = mat[swap_1st, :]
	return new_mat


def get_perm_matrix_swap(num_perspectives, first_half_list, second_half_list):
	mat = torch.eye(2*num_perspectives, 2*num_perspectives) 
	new_mat = mat.clone()
	for f, s in zip(first_half_list, second_half_list):
		new_mat[f, :] = mat[s, :]; new_mat[s, :] = mat[f, :]
	return new_mat

def get_avg_matrix(num_perspectives):
	mat = torch.zeros(2*num_perspectives, 2)
	for i in range(num_perspectives):
		mat[i, 0] = 1
		mat[i+num_perspectives, 1]=1
	return (1/num_perspectives) * mat

def put_labels(batch_size, zero_tensor):
	assert len(zero_tensor) %2 ==0
	for i in range(len(zero_tensor)):
		zero_tensor[i] = i % batch_size
	return zero_tensor


def put_mask(batch_size, zero_mat):
	#Put mask for the first "batch_size" elements always
	assert len(zero_tensor) %batch_size ==0
	for i in range(len(zero_tensor)):
		zero_tensor[i] = i % batch_size
	return zero_tensor

#Make sure that the input tensor is 1d and has shape e.g. torch.Size([6])
def shift(input_1d_tensor, shift_by):
	output_tensor = torch.zeros(input_1d_tensor.shape, dtype=input_1d_tensor.dtype)
	where = torch.where(input_1d_tensor)[0]
	output_tensor[where+shift_by] = input_1d_tensor[where]
	return output_tensor


def get_mask_logits(M, batch_size, num_permutations=1):
	mask = torch.zeros(2*M*num_permutations, M*num_permutations)
	default_1d_tensor = torch.zeros(M*num_permutations).long()
	default_1d_tensor[:batch_size*num_permutations] = 1
	assert M*num_permutations % (2*num_permutations) ==0
	for i in range(int(M/(2))):
		mask[i*4*num_permutations:(i+1)*4*num_permutations,:] =  shift(default_1d_tensor, batch_size*num_permutations*i)
	return mask

def shuffle(logits, labels, mask_logits, seed):
	#Shuffle over 2*M
	np.random.seed(seed)
	assert logits.shape[0] == labels.shape[0]
	assert mask_logits.shape[0] == labels.shape[0]
	permuted = np.random.permutation(logits.shape[0]).tolist()
	return logits[permuted], labels[permuted], mask_logits[permuted] 


def nll(logits, mask_logits, labels, minus_no, usual_nll=False):
	summed = torch.sum(torch.exp(logits * (1-mask_logits)), axis=1) - minus_no #2 #mask logits 0 되는 곳 exp 하면 1 되는게 문제임
	#summed = torch.sum(torch.exp(logits ), axis=1) 
	label_logits = logits[range(logits.shape[0]), labels.tolist()]
	if usual_nll:
		summed = summed + torch.exp(label_logits)
	#softmaxed = torch.exp(label_logits - torch.log(summed))
	log_softmaxed = -(label_logits - torch.log(summed))
	return torch.mean(log_softmaxed)

def get_max_logit(logits):
	#logits should be 1 d
	assert logits.shape[0] %3 ==0
	max_logits = []
	for i in range(int(logits.shape[0]/3)):
		max_logits.append(max(logits[3*i:3*(i+1)]).detach().cpu().item())
	return max_logits

def get_labels(batch_size, num_classes, num_permutations=1):
	labels = []
	for i in range(num_classes):
		for j in range(batch_size):
			nrange = np.arange(num_permutations*batch_size) + i*num_permutations*batch_size
			labels += nrange.tolist()
	return labels


import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def p_value(subset_logits):
	#p = 
	#return p
	pass

#Takes in logits of shape (num_classes**2,train_batch_size)  or (num_classes**2, (self.args.permclr_views**2+1)*train_batch_size))
#and outputs
#p-values (new logits) of shape
#num_classes**2 


def tpr():
	pass

def tnr():
	pass




class PermCLR(object):
	def __init__(self, *args, **kwargs):
		self.args = kwargs['args']
		self.model = kwargs['model'].to(self.args.device)
		self.optimizer = kwargs['optimizer']
		self.scheduler = kwargs['scheduler']
		self.writer = SummaryWriter()
		logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
		self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

	def classifier(self, logits, train_batch_size, num_permutations, num_classes, indicator=False):
		#Define original and permutation
		#In the 0th axis, 0, 4, 8 (which are 0, middle in 2nd row, last) are the same class
		#In the 1st axis, the first train_batch_size are the "T" of the originals (identity permutation)
		
		#Get the T of the originals (identity permutation) <- We already have these

		#Compute the difference of the permutations (the rest (self.args.permclr_views**2)* train_batch_size of axis 1) with these T
		total_minus = torch.zeros(logits.shape[0], train_batch_size).to(self.args.device)
		total_minus_save = torch.cat([total_minus] * (num_permutations-1), axis=1)
		for i in range(num_permutations-1):
			#Count instances larger than the original
			minus = logits[:, (1+i)*train_batch_size : (2+i)*train_batch_size] - logits[:, 0 : train_batch_size]
			#minus = minus >0 
			total_minus_save[:, (i)*train_batch_size : (1+i)*train_batch_size] = minus
			total_minus += minus

		total_minus = total_minus/ (num_permutations-1)
		#print("total_minus is ", total_minus)

		#Maybe - Average over train_batch_size
		if not(indicator):
			#print("total_minus is ", total_minus)
			new_logits = torch.mean(torch.abs(total_minus), axis=1) #shape is logits.shape[0]
			new_logits = 1 - new_logits
		else:
			#print("total_minus is ", total_minus)
			#Count indicator across column (smallest among 0,1,2/3,4,5/6,7,8)
			new_logits = torch.zeros(logits.shape[0])
			#new_logits = torch.zeros(total_minus.shape)
			#argmins = torch.argmin(total_minus, axis=0)
			#for a in argmins.cpu().tolist():
			#convert to numpy and try
			#total_minus = total_minus.detach().cpu().numpy()
			#total_minus = torch.abs(total_minus)
			#print("total minus is ", total_minus)
			total_minus_no_abs = copy.deepcopy(total_minus.detach().cpu().numpy())
			total_minus = torch.abs(total_minus)

			#Get the 1/3 threshold for each of  0,1,2/3,4,5/6,7,8
			for i in range(num_classes):
				sorted, indices = torch.sort(total_minus.view(-1)[i*(num_classes*train_batch_size):(i+1)*(num_classes*train_batch_size)])
				threshold = sorted[train_batch_size-1] #The 1/3th smallest
				#Count how many of the indices before 10 belongs to each of the three classes
				indices = indices[:train_batch_size]
				for j in range(num_classes):
					new_logits[i*(num_classes) + j ] = torch.sum((train_batch_size*j<=indices) * (indices<train_batch_size*(j+1))).float()/train_batch_size
			#print("new new_logitts are ", new_logits)

			#wheres = torch.cat([torch.arange(logits.shape[0]).unsqueeze(0), argmins.unsqueeze(0)], axis=0).T
			#new_logits[wheres] = 1

		#Maybe filter what to take in later

		#Or use the fact that p-value itself is uniform?

		return new_logits, total_minus_save #large is bad

	#For test and ood
	#def inference(self, train_datasets, test_datasets, train_loaders, test_loaders, f, just_average=True, num_train_batch=1):
	def inference(self, train_datasets, test_datasets, test_loaders, f, just_average=True, train_batch_size=1, p_classifier=False, get_cutoff=False):
		torch.cuda.set_device(0)
		num_classes = len(train_datasets)
		scaler = GradScaler(enabled=self.args.fp16_precision) 

		if get_cutoff:
			logit_list = []

		num_perms = 6*6 + 1 #4c2 * 4c2  +1 

		auroc_max_logits = []
		auroc_labels = []
		class_lens = [len(td) for td in train_datasets]

		#ORIGINAL OF JUST AVG BRANCH
		if just_average:
			P_mat = get_perm_matrix_identity(self.args.permclr_views).to(self.args.device) #has shape 8x8 
			P_mat_128 = torch.cat([P_mat.unsqueeze(0)]*128, axis=0).float()

		else:
			P_mats = [torch.eye(2*self.args.permclr_views).to(self.args.device)]
			#list of combinations
			#comb = itertools.combinations(np.arange(self.args.permclr_views).tolist(), 2)
			for subset in itertools.combinations(np.arange(self.args.permclr_views).tolist(), 2):
				for subset_two in itertools.combinations(np.arange(self.args.permclr_views).tolist(), 2):
					subset_two_new = tuple([i+self.args.permclr_views for i in subset_two])
					P_mats.append(get_perm_matrix_swap(self.args.permclr_views, subset, subset_two_new).to(self.args.device))
			#print("len Pmats is ", len(P_mats))
			P_mat = torch.block_diag(*P_mats) #Has shape torch.Size([136, 136]) (4*2) * (4**2+1) or (args.permclr_views * batch_size) * (args.permclr_views**2 + 1)
			del P_mats
			P_mat_128 = torch.cat([P_mat.unsqueeze(0)]*128, axis=0).float() #Has shape 

		avg_matrix = get_avg_matrix(self.args.permclr_views) #8x2
		avg_matrix_128 = torch.cat([avg_matrix.unsqueeze(0)]*128, axis=0).to(self.args.device)

		#Sample train examples once before the loop

		chosens = []
		for ci, c in enumerate(class_lens):
			np.random.seed(1000*ci)
			#Just choose one
			chosens.append(np.random.choice(c, train_batch_size).tolist())

		train_category_labels_tup =[]
		for ci, chosen in enumerate(chosens):
			cat_by_category = []
			for c in chosen:
				batch_dict = train_datasets[ci][c] 
				cat_by_category += [batch_dict['image_' + str(i)].unsqueeze(0) for i in range(self.args.permclr_views)]
			catted_imgs = torch.cat(cat_by_category)
			train_category_labels_tup.append(catted_imgs)

		for batch_i, batch_dict_tuple in enumerate(zip(*test_loaders)): 
			#Just know how many objects per class there are in this batch
			#cur_batch_size = batch_dict_tuple[0]['image_0'].shape[0] #TODO later
			#Get a random object from each category of train_dataset
			
			#chosen training
			# chosen_train_dataset_tuples = [train_datasets[i][chosens[i]] for i in range(len(train_datasets))] 
			# pickle.dump(chosen_train_dataset_tuples, open('chosen_train_dataset_tuple_b1.p', 'wb'))
			# #catted_img_tups of train dataset
			# train_category_labels_tup =[]
			# for batch_dict in chosen_train_dataset_tuples:
			# 	catted_imgs = torch.cat([batch_dict['image_' + str(i)].unsqueeze(0) for i in range(self.args.permclr_views)]) 
			# 	train_category_labels_tup.append(catted_imgs) #each catted_image has shape torch.Size([self.args.permclr_views, 3, 32, 32])]

			

			#catted_img_tups of test dataset
			catted_imgs_tup = []
			object_labels_tup = []
			category_labels_tup =[]
			#concatente all the image_i's together in one direction(image_0: all the image_0's, image_3's: all the image_3's)
			for batch_dict in batch_dict_tuple:
				catted_imgs = torch.cat([batch_dict['image_' + str(i)] for i in range(self.args.permclr_views)]) #shape is torch.Size([8, 3, 32, 32]) #8 is batch_size * num_objects (permclr_views)
				if not(self.args.ood):
					object_labels = torch.cat([batch_dict['object_label'] for i in range(self.args.permclr_views)]) #shape is torch.Size([8])
					category = self.args.classes_to_idx[batch_dict['category_label'][0]]
					category_labels_tup.append(torch.tensor([category]*self.args.permclr_views*self.args.batch_size))
					object_labels_tup.append(object_labels)
				catted_imgs_tup.append(catted_imgs)

			#Concatenate everything into batch_imgs
			batch_imgs = torch.cat(train_category_labels_tup + catted_imgs_tup) # shape is torch.Size([self.args.permclr_views* (batch_size * num_classes + num_classes*train_batch_size), 3, 32, 32]) #The first self.args.permclr_views * num_classes are train imags
			batch_imgs = batch_imgs.to(self.args.device)

			#Put into model and get features
			with autocast(enabled=self.args.fp16_precision):
				features = self.model(batch_imgs) #shape is torch.Size([self.args.permclr_views* (batch_size * num_classes + num_classes), 128])

			#Now separate into two
			features_train = features[:self.args.permclr_views*num_classes*train_batch_size, :].clone() #shape  torch.Size([self.args.permclr_views*  num_classes, 128])
			features_test = features[self.args.permclr_views*num_classes*train_batch_size:, :].clone() #shape torch.Size([self.args.permclr_views* (batch_size * num_classes), 128])
			del features

			#Stack and concatenate
			#Stack features_train first
			features_train = features_train.reshape(num_classes*train_batch_size, self.args.permclr_views, -1)
			features_train = torch.cat([features_train]*num_classes) #ASSUME BATCH_SIZE=1 #CHANGE FROM HERE IF CHANGE BATCH SIZE #shape should be (num_classes**2*train_batch_size, self.args.permclr_views, 128 )

			#Stack features_test
			features_test = features_test.reshape(num_classes, self.args.permclr_views, -1) #ASSUME BATCH_SIZE=1
			features_test = features_test.transpose(0,1) #(self.args.permclr_views, num_classes, 128)
			features_test = torch.cat([features_test]*num_classes*train_batch_size) #(self.args.permclr_views*num_classes, num_classes, 128)
			features_test = features_test.transpose(0,1)
			features_test = features_test.reshape(num_classes**2*train_batch_size, self.args.permclr_views, -1) #shape is (num_classes**2*train_batch_size,, self.args.permclr_views, 128 ) WITH batch size 1

			#Concatenate feature_test and features_train 
			features = torch.cat([features_test, features_train], axis=1) #shape is (num_classes**2*train_batch_size,, self.args.permclr_views*2, 128 ) WITH batch size 1

			#Permute
			#Reshape features for permuting
			#Reshaped into (128, num_classes**2,  self.args.permclr_views*2)

			if not(just_average):
				features = torch.cat([features]*(num_perms), axis=1)#Shape is 9 x (8*17) x 128
			features = features.permute(2, 1, 0) #Now shape is 128 x self.args.permclr_views*2x num_classes**2 (used to be 36 x 8x 128)
			#print("P mat 128 shape", P_mat_128.shape)
			#print("features shape", features.shape)
			features = torch.bmm(P_mat_128, features) #shape is still 128, 8, 9 
			features = features.permute(0, 2, 1) #Shape is now 128 x 9 x 8. THIS IS (kind of? reshaped) THE PERMUTED B (B * P^T)

			if not(just_average):
				features = features.reshape(self.args.out_dim, (num_classes**2) * (num_perms)*train_batch_size, self.args.permclr_views*2)  #shape (128, 9*17, 8)

			#Get average
			features = torch.bmm(features, avg_matrix_128) #This is the average features in Part2-2 #Shape is torch.Size([128, 9, 2]) or torch.Size([128, 9*17, 2])


			#Take dot product
			#Normalize across 128
			features[:, :, 0] = F.normalize(features[:, :, 0].clone(), dim=0); features[:, :, 1] = F.normalize(features[:, :, 1].clone(), dim=0)
			#Multiply elementwise among the dimension of "2"
			features = torch.mul(features[:, :, 0].clone(), features[:,:,1].clone()) #Shape is torch.Size([128, 9]) or torch.Size([128, 9*17])
			#Now sum across the 128 dimensions
			logits = torch.sum(features, axis=0) #Shape is torch.Size([9*train_batch_size]) or torch.Size([9*train_batch_size*17])

			#For sanity check just average over train_batch_size
			if (just_average):
				logits = logits.reshape(num_classes**2,train_batch_size) #The first tranin_batchsize are car_test, car_train(1,2,..,train_batch_size,), ...
				logits = torch.mean(logits, axis=1)


			#If not just_average, reshape logits and get avg
			if not(just_average):
				logits = logits.reshape(num_classes**2, (num_perms)*train_batch_size) # The firsr 17*train_batch_size is car_test*car_train, the 2nd 17*train_batch_size is car_test*bat_train, .., the fourth 17 is bat_test*car_train, ...
				if not(p_classifier):
					#Average across axis 1 (across the 17)
					logits = torch.mean(logits, axis=1)
				else:
					logits, total_minus_save = self.classifier(logits, train_batch_size, self.args.permclr_views,  num_classes, self.args.indicator_classifier)

			total_minus_save = total_minus_save.detach().cpu()
			if get_cutoff:
				logit_list.append(total_minus_save)

			logits= logits.detach().cpu().numpy()
			#logits = logits.tolist()
			#Save the max of logits for each example 

			#Print logits into file
			assert logits.shape[0] %3 ==0
			auroc_max_logits += get_max_logit(logits)
			if not(self.args.ood):
				auroc_labels+= [1] * int(logits.shape[0]/3)
			else:
				auroc_labels+= [0] * int(logits.shape[0]/3)
			f.write("logits for batch :" + str(batch_i) + '\n')
			f.write(str(logits) + '\n')
		f.close()
		pickle.dump(logit_list, open('logits/cutoff_test.p', 'wb'))
		return auroc_max_logits, auroc_labels


	def train(self, train_datasets, train_loaders, debug_with_identity=False):
		
		torch.cuda.set_device(0)
		num_classes = len(train_loaders)
		scaler = GradScaler(enabled=self.args.fp16_precision)

		A_mat = get_A_matrix(self.args.permclr_views) #8*8
		A_mat = torch.block_diag(*[A_mat]*len(self.args.classes_to_idx)).to(self.args.device) #24x24 with Car1, Car2, Cat1, Cat2, ...

		#ORIGINAL OF JUST AVG BRANCH
		#P_mat = get_perm_matrix_identity(self.args.permclr_views).to(self.args.device) #has shape 8x8 
		#P_mat_128 = torch.cat([P_mat.unsqueeze(0)]*128, axis=0).float()

		num_permutations = self.args.num_perms**2 + 1

		#Control here with num_permutations
		P_mats = [torch.eye(2*self.args.permclr_views).to(self.args.device)]
		for i in range(self.args.num_perms):
			for j in range(self.args.num_perms):
				if not(debug_with_identity):
					P_mats.append(get_perm_matrix_one(self.args.permclr_views, i, self.args.permclr_views+j).to(self.args.device))
				else:
					P_mats.append(get_perm_matrix_identity(self.args.permclr_views).to(self.args.device))
		P_mat = torch.block_diag(*P_mats) #Has shape torch.Size([136, 136]) (4*2) * (4**2+1) or (args.permclr_views * batch_size) * (args.permclr_views**2 + 1)
		del P_mats
		P_mat_128 = torch.cat([P_mat.unsqueeze(0)]*128, axis=0).float() #Has shape (128 x 136 x 136) 

		avg_matrix = get_avg_matrix(self.args.permclr_views) #8x2
		avg_matrix_128 = torch.cat([avg_matrix.unsqueeze(0)]*128, axis=0).to(self.args.device)

		for epoch_counter in range(self.args.epochs):
			mean_loss = 0.0
			for batch_i, batch_dict_tuple in enumerate(zip(*train_loaders)): 
				if batch_i % 20==0:
					print("batch i is ", batch_i)

				catted_imgs_tup = []
				object_labels_tup = []
				category_labels_tup =[]
				for batch_dict in batch_dict_tuple:
					catted_imgs = torch.cat([batch_dict['image_' + str(i)] for i in range(self.args.permclr_views)]) #shape is torch.Size([8, 3, 32, 32]) #8 is batch_size * num_objects (permclr_views)
					object_labels = torch.cat([batch_dict['object_label'] for i in range(self.args.permclr_views)]) #shape is torch.Size([8])
					category = self.args.classes_to_idx[batch_dict['category_label'][0]]
					category_labels_tup.append(torch.tensor([category]*self.args.permclr_views*self.args.batch_size))
					catted_imgs_tup.append(catted_imgs); object_labels_tup.append(object_labels)
				
				batch_imgs = torch.cat(catted_imgs_tup) #Shape should be like torch.Size([24, 3, 32, 32])
				del batch_dict, catted_imgs_tup, catted_imgs
				batch_object_labels = torch.cat(object_labels_tup) #Shape should be like torch.Size([24])
				del object_labels_tup, object_labels
				batch_category_labels = torch.cat(category_labels_tup) #Shape should be like torch.Size([24])

				#send to device
				batch_imgs = batch_imgs.to(self.args.device)
				batch_object_labels = batch_object_labels.to(self.args.device)
				batch_category_labels = batch_category_labels.to(self.args.device)

				#PART1
				#1. Put all of the images into a model and get features
				with autocast(enabled=self.args.fp16_precision):
					features = self.model(batch_imgs) #Shape should be like torch.Size([24, 128])

				#2. Rearrange these features (A) #M=  batch_size * num_categories (e.g. 6 in this case where there are 3 classes)
					features = torch.mm(A_mat, features) 
					#batch_category_labels = torch.mm(A_mat, batch_category_labels.view(1,-1).T.float()).long()
					#batch_object_labels = torch.mm(A_mat, batch_object_labels.view(1,-1).T.float()).long()

					features =features.reshape(self.args.batch_size*len(self.args.classes_to_idx), self.args.permclr_views, -1)  #THIS is A
					#batch_category_labels = batch_category_labels.squeeze().reshape(self.args.batch_size*len(self.args.classes_to_idx), self.args.permclr_views)
					#batch_object_labels = batch_object_labels.squeeze().reshape(self.args.batch_size*len(self.args.classes_to_idx), self.args.permclr_views)
					#print("gpu memory after A: ", get_gpu_memory())
					
				#3. Concatenate (B)
					M = self.args.batch_size*len(self.args.classes_to_idx)
					features = torch.cat([torch.cat([features]*M, axis = 1).reshape(M**2,self.args.permclr_views,-1), torch.cat([features]*M)], axis=1)
					#batch_category_labels = torch.cat([torch.cat([batch_category_labels]*M, axis = 1).reshape(M**2,self.args.permclr_views), torch.cat([batch_category_labels]*M)], axis=1)
					#batch_object_labels = torch.cat([torch.cat([batch_object_labels]*M, axis = 1).reshape(M**2,self.args.permclr_views), torch.cat([batch_object_labels]*M)], axis=1)
					#print("gpu memory after B: ", get_gpu_memory())

					#Concatenate B horizontally args.permclr_views **2 + 1 (Page 5 beginning)
					features = torch.cat([features]*(num_permutations), axis=1)#Shape is 36 x (8*17) x 128
					#Shape becomes 128 x (8*17) x 36 with features = features.permute(2, 1, 0) below.

				#4. Permute (B P^T)
					#Multiply by a permutation matrix for each row
					#batch_category_labels = torch.mm(batch_category_labels.float(),P_mat.T) #Shape is 36x8
					#batch_object_labels = torch.mm(batch_object_labels.float(),P_mat.T)
					features = features.permute(2, 1, 0) #Now shape is 128 x 8x 36 (used to be 36 x 8x 128)
					features = torch.bmm(P_mat_128, features) #shape is 128, 8, 36 
					features = features.permute(0, 2, 1) #Shape is now 128 x 36 x (8*17). THIS IS (kind of? reshaped) THE PERMUTED B (B * P^T)
					features = features.reshape(self.args.out_dim, (M**2) * (num_permutations), self.args.permclr_views*self.args.batch_size) #shape (128, 36*17, 8)
					#features = features.transpose(1,2) #shape is (128, 8, 36*17) #IS THIS THE RIGHT SHAPE? THIS is not needed.


				#PART2
				#1. Get average features
					features = torch.bmm(features, avg_matrix_128) #This is the average features in Part2-2 #Shape is torch.Size([128, 36*17, 2])
					

				#2. Get score matrix
					features[:, :, 0] = F.normalize(features[:, :, 0].clone(), dim=0); features[:, :, 1] = F.normalize(features[:, :, 1].clone(), dim=0)
					#torch.mul for elementwise multiplication of matrices
					features = torch.mul(features[:, :, 0].clone(), features[:,:,1].clone()) #Shape is torch.Size([128, 36*17])
					#Now sum across the 128 dimensions
					logits = torch.sum(features, axis=0) #Shape is torch.Size([36*17])

				#3. Put this into NLL loss
					#Make logits into torch.Size([M x M]) (e.g. 6x6)
					logits = logits.reshape(M, (num_permutations)*M)
					#Positives are the first "batch_size" of each row in the [6x6] above (which has size batch_size * num_classes(M))
					#copy into (batch_size * M) x M
					logits = torch.cat([logits.T]*(self.args.batch_size*num_permutations), axis=0).T.reshape((self.args.batch_size*num_permutations)*M, M*num_permutations) #torch.Size([12, 6])
					pickle.dump(logits, open("logits.p", "wb"))
					#Get labels for logits
					#Change this later if batchsize, # classes or anything is changed!
					#labels = torch.tensor([0,1,0,1,2,3,2,3,4,5,4,5]).to(self.args.device)
					labels = torch.tensor(get_labels(self.args.batch_size, num_classes, num_permutations)).to(self.args.device)
					#Mask logits so that the positives are not counted (e.g. for row 0, 1 is the mask)
					mask_logits = get_mask_logits(M, self.args.batch_size, num_permutations).to(self.args.device)
					#Code NLL loss with ignore indices
					logits = logits / self.args.temperature
					#Shuffle everything before putting into nll
					logits, labels, mask_logits = shuffle(logits, labels, mask_logits, batch_i + 100*epoch_counter)
					loss = nll(logits, mask_logits, labels, self.args.batch_size*num_permutations, self.args.usual_nll)
					mean_loss += loss.detach().cpu().item()

				#Optimizer zero grad
					self.optimizer.zero_grad()

					scaler.scale(loss).backward()

					scaler.step(self.optimizer)
					scaler.update()

				#Schedule
			if epoch_counter >= 10:
				self.scheduler.step()
			print("Epoch: " + str(epoch_counter) +"Mean Loss: " + str(mean_loss/ (batch_i+1)))
			print("Epoch: " + str(epoch_counter) +"Loss: " + str(loss))


			#shuffle
			for i in range(num_classes):
				train_datasets[i].shuffle()
				train_loaders[i] = torch.utils.data.DataLoader(train_datasets[i], batch_size=self.args.batch_size,num_workers=self.args.workers, pin_memory=True, drop_last=True)
				