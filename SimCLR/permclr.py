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

def nll(logits, mask_logits, labels):
	summed = torch.sum(torch.exp(logits * mask_logits), axis=1)
	label_logits = logits[range(logits.shape[0]), labels.tolist()]
	softmaxed = torch.exp(label_logits - torch.log(summed))
	return torch.mean(softmaxed)

import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values




class PermCLR(object):
	def __init__(self, *args, **kwargs):
		self.args = kwargs['args']
		self.model = kwargs['model'].to(self.args.device)
		self.optimizer = kwargs['optimizer']
		self.scheduler = kwargs['scheduler']
		self.writer = SummaryWriter()
		logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
		self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

	def train(self, train_datasets, train_loaders):
		torch.cuda.set_device(0)
		classes = len(train_loaders)
		scaler = GradScaler(enabled=self.args.fp16_precision)

		for epoch_counter in range(self.args.epochs):
			for batch_dict_tuple in zip(*train_loaders): 
				#batch_dict_tuple is a tuple of batch_dict's
				#batch_dict_tuple[0]['image_0'] has shape torch.Size([2, 3, 32, 32]) if batch_size is 2
				catted_imgs_tup = []
				object_labels_tup = []
				category_labels_tup =[]
				#concatente all the image_i's together in one direction(image_0: all the image_0's, image_3's: all the image_3's)
				for batch_dict in batch_dict_tuple:
					catted_imgs = torch.cat([batch_dict['image_' + str(i)] for i in range(self.args.permclr_views)]) #shape is torch.Size([8, 3, 32, 32]) #8 is batch_size * num_objects (permclr_views)
					object_labels = torch.cat([batch_dict['object_label'] for i in range(self.args.permclr_views)]) #shape is torch.Size([8])
					category = self.args.classes_to_idx[batch_dict['category_label'][0]]
					category_labels_tup.append(torch.tensor([category]*self.args.permclr_views*self.args.batch_size))
					#class_labels = torch.cat([args.classes_to_idx[batch_dict['category_label']] for i in range(self.args.permclr_views)]) 
					catted_imgs_tup.append(catted_imgs); object_labels_tup.append(object_labels)
				
				#collapse all the "i"'s together in another direction
				batch_imgs = torch.cat(catted_imgs_tup) #Shape should be like torch.Size([24, 3, 32, 32])
				del batch_dict, catted_imgs_tup, catted_imgs
				batch_object_labels = torch.cat(object_labels_tup) #Shape should be like torch.Size([24])
				del object_labels_tup, object_labels
				batch_category_labels = torch.cat(category_labels_tup) #Shape should be like torch.Size([24])

				#send to device
				batch_imgs = batch_imgs.to(self.args.device)
				batch_object_labels = batch_object_labels.to(self.args.device)
				batch_category_labels = batch_category_labels.to(self.args.device)

				pickle.dump(batch_imgs, open("batch_imgs.p", "wb"))
				pickle.dump(batch_object_labels, open("batch_object_labels.p", "wb"))
				pickle.dump(batch_category_labels, open("batch_category_labels.p", "wb"))
				pickle.dump(self.args, open("args.p", "wb"))
				#PART1
				#1. Put all of the images into a model and get features
				with autocast(enabled=self.args.fp16_precision):
					features = self.model(batch_imgs) #Shape should be like torch.Size([24, 128])
					print("gpu memory after features: ", get_gpu_memory())
					pickle.dump(features, open("features.p", "wb"))

				#2. Rearrange these features (A) #M=  batch_size * num_categories (e.g. 6 in this case where there are 3 classes)
					#features = features.reshape(self.args.batch_size * len(self.args.classes_to_idx), self.args.permclr_views, -1)
					#TODO: check if this is correct
					#Copy this matrix diagonally and apply it to batch_object_labels
					A_mat = get_A_matrix(self.args.permclr_views) #8*8
					A_mat = torch.block_diag(*[A_mat]*len(self.args.classes_to_idx)).to(self.args.device) #24x24 with Car1, Car2, Cat1, Cat2, ...
					features = torch.mm(A_mat, features) #Now we are ready to reshape this and make "A". Reshaping this is "A".
					pickle.dump(A_mat, open("A_mat1.p", "wb"))
					pickle.dump(features, open("features1.p", "wb"))
					batch_category_labels = torch.mm(A_mat, batch_category_labels.view(1,-1).T.float()).long()
					batch_object_labels = torch.mm(A_mat, batch_object_labels.view(1,-1).T.float()).long()

					features =features.reshape(self.args.batch_size*len(self.args.classes_to_idx), self.args.permclr_views, -1)  #THIS is A
					batch_category_labels = batch_category_labels.squeeze().reshape(self.args.batch_size*len(self.args.classes_to_idx), self.args.permclr_views)
					batch_object_labels = batch_object_labels.squeeze().reshape(self.args.batch_size*len(self.args.classes_to_idx), self.args.permclr_views)
					print("gpu memory after A: ", get_gpu_memory())
					
				#3. Concatenate (B)
					M = self.args.batch_size*len(self.args.classes_to_idx)
					#features = torch.cat([torch.cat([features]*6, axis = 1).reshape(36,4,-1), torch.cat([features]*6)], axis=1)
					features = torch.cat([torch.cat([features]*M, axis = 1).reshape(M**2,self.args.permclr_views,-1), torch.cat([features]*M)], axis=1)
					batch_category_labels = torch.cat([torch.cat([batch_category_labels]*M, axis = 1).reshape(M**2,self.args.permclr_views), torch.cat([batch_category_labels]*M)], axis=1)
					batch_object_labels = torch.cat([torch.cat([batch_object_labels]*M, axis = 1).reshape(M**2,self.args.permclr_views), torch.cat([batch_object_labels]*M)], axis=1)
					print("gpu memory after B: ", get_gpu_memory())

				#4. Permute (B P^T)
					#Multiply by a permutation matrix for each row
					#P_mat = torch.zeros(self.args.permclr_views*self.args.batch_size, self.args.permclr_views*self.args.batch_size).cuda()
					P_mat = get_perm_matrix(self.args.permclr_views) #has shape 8x8 
					#torch.cat([torch.cat([batch_category_labels]*6, axis = 1).reshape(-1,4), torch.cat([batch_category_labels]*6)], axis=1).shape
					batch_category_labels = torch.mm(batch_category_labels.float(),P_mat.T) #Shape is 36x8
					batch_object_labels = torch.mm(batch_object_labels.float(),P_mat.T)
					#Apply bmm
					#torch.cuda.empty_cache()
					P_mat_128 = torch.cat([P_mat.unsqueeze(0)]*128, axis=0).float().to(self.args.device)
					#P_mat_128 = P_mat_128.to(torch.device("cuda:1")) #Now shape is 128x8x8
					pickle.dump(P_mat_128, open("P_mat_128.p", "wb"))
					features = features.permute(2, 1, 0) #Now shape is 128 x 8x 36 (used to be 36 x 8x 128)
					pickle.dump(features, open("features4.p", "wb"))
					features = torch.bmm(P_mat_128, features) #shape is 128, 8, 36 
					features = features.permute(0, 2, 1) #Shape is now 128 x 36 x 8. THIS IS (kind of? reshaped) THE PERMUTED B (B * P^T)

				#PART2
				#1. Get average features
					avg_matrix = get_avg_matrix(self.args.permclr_views) #8x2
					#avg_matrix_128 = torch.zeros(128, self.args.permclr_views*self.args.batch_size, self.args.batch_size).float().to(self.args.device)
					avg_matrix_128 = torch.cat([avg_matrix.unsqueeze(0)]*128, axis=0)
					avg_matrix_128 = avg_matrix_128.to(self.args.device) #torch.Size([128, 8, 2])
					features = torch.bmm(features, avg_matrix_128) #This is the average features in Part2-2 #Shape is torch.Size([128, 36, 2])

				#2. Get score matrix
					#Normalize before the elementwise multiplication, for cosine similarity
					#Normalize across 128
					features[:, :, 0] = F.normalize(features[:, :, 0], dim=0); features[:, :, 1] = F.normalize(features[:, :, 1], dim=0)
					#Multiple among the dimension of "2"
					#Shape should be 128 x 36
					#torch.mul for elementwise multiplication of matrices
					features = torch.mul(features[:, :, 0], features[:,:,1]) #Shape is torch.Size([128, 36])
					#Now sum across the 128 dimensions
					logits = torch.sum(features, axis=0) #Shape is torch.Size([36])

				#3. Put this into NLL loss
					#Make logits into torch.Size([M x M]) (e.g. 6x6)
					logits = logits.reshape(M, M)
					#Positives are the first "batch_size" of each row in the [6x6] above (which has size batch_size * num_classes(M))
					#copy into (batch_size * M) x M
					logits = torch.cat([logits.T]*self.args.batch_size, axis=0).T.reshape(2*M, M) #torch.Size([12, 6])
					#Get labels for logits
					labels = torch.zeros(logits.shape[0], dtype=torch.long)	
					labels = put_labels(self.args.batch_size, labels)
					#Mask logits so that the positives are not counted (e.g. for row 0, 1 is the mask)
					mask_logits = torch.cat([torch.ones(logits.shape[0], self.args.batch_size), torch.zeros(logits.shape[0], logits.shape[1] - self.args.batch_size)], axis=1)	
					#Code NLL loss with ignore indices
					logits = logits / self.args.temperature
					loss = nll(logits, mask_logits, labels)

				#Optimizer zero grad
					self.optimizer.zero_grad()

					scaler.scale(loss).backward()

					scaler.step(self.optimizer)
					scaler.update()

				#Schedule
			if epoch_counter >= 10:
				self.scheduler.step()
			print("Epoch: " + str(epoch_counter) +"Loss: " + str(loss))




			#shuffle
			for i, c in enumerate(classes):
				train_datasets[i].shuffle()
				train_oaders[i] = torch.utils.data.DataLoader(train_datasets[i], batch_size=args.batch_size,num_workers=args.workers, pin_memory=True)
				