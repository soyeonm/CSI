#Maybe do like Permclr but use SupContrast loss

import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy

import pickle
import time
import numpy as np
import copy

import itertools
from permclr import get_perm_matrix_identity, get_avg_matrix, get_max_logit

import torch.distributed as dist
import diffdist.functional as distops
import datetime
torch.manual_seed(0)


def save_checkpoint(epoch, model, save_name, save_dir, multi_gpu, rank=0):
    last_model = os.path.join(save_dir, save_name+ "_epoch_" + str(epoch))
    if rank!=0:
    	last_model = os.path.join(save_dir, save_name+ "rank_" + str(rank) + "_epoch_" + str(epoch))
    if multi_gpu:
        torch.save(model.module.state_dict(), last_model)
    else:
        torch.save(model.state_dict(), last_model)
    

def get_max_logit_refactored_march10(logits, labels, num_classes):
	#logits should be 1 d
	assert logits.shape[0] %num_classes ==0
	assert logits.shape[0]/ num_classes == labels.shape[0]
	max_logits = []
	argmax_aligns = []
	for i in range(int(logits.shape[0]/num_classes)):
		#print("logit is ", logits[3*i:3*(i+1)])
		max_logits.append(max(logits[num_classes*i:num_classes*(i+1)]))
		argmax_aligns.append(np.argmax(logits[num_classes*i:num_classes*(i+1)]) == labels[i])
		#print("argmax for i: ", i, " is ", np.argmax(logits[3*i:3*(i+1)]))
		#print("argmax aligns last element is ", argmax_aligns[-1])
	return max_logits, argmax_aligns

class ObjCLR(object):
	def __init__(self, temperature=0.07, contrast_mode='all',
				 base_temperature=0.07, *args, **kwargs):
		self.args = kwargs['args']
		self.model = kwargs['model']#.to(self.args.device)
		self.optimizer = kwargs['optimizer']
		self.scheduler = kwargs['scheduler']
		#self.writer = SummaryWriter()
		#logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
		self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)


		self.temperature = temperature
		self.contrast_mode = contrast_mode
		self.base_temperature = base_temperature

	def sup_con_loss(self, features, labels=None, mask=None, multi_gpu = False):
		device = features.device #device = self.args.device
		if len(features.shape) < 3: 
			raise ValueError('`features` needs to be [bsz, n_views, ...],'
							 'at least 3 dimensions are required')
		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		#####################Multigpu
		if multi_gpu:
			features_gathered = []
			for out in features.chunk(chunk):
				gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
				gather_t = torch.cat(distops.all_gather(gather_t, out))
				features_gathered.append(gather_t)
			outputs = torch.cat(features_gathered)
			print("shape of features is ", features.shape)

		#What do I do about this?
		if multi_gpu and not(labels is None):	
			gather_t = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
			labels = torch.cat(distops.all_gather(gather_t, labels))
		# Then what do I do about masks? -> automatically done
		#Set every "device" to features' device -> done
		#####################


		batch_size = features.shape[0]
		if labels is not None and mask is not None:
			raise ValueError('Cannot define both `labels` and `mask`')
		elif labels is None and mask is None:
			mask = torch.eye(batch_size, dtype=torch.float32).to(device)
		elif labels is not None:
			labels = labels.contiguous().view(-1, 1)
			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')
			mask = torch.eq(labels, labels.T).float().to(device)
		else:
			mask = mask.float().to(device)


		contrast_count = features.shape[1] #2
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #Flattens into 200, 128
		if self.contrast_mode == 'one':
			anchor_feature = features[:, 0]
			anchor_count = 1
		elif self.contrast_mode == 'all':
			anchor_feature = contrast_feature
			anchor_count = contrast_count
		else:
			raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(anchor_feature, contrast_feature.T), #this must be the dot product of all entries to each other
			self.args.temperature)
		# for numerical stability
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach() 

		# tile mask
		mask = mask.repeat(anchor_count, contrast_count) #mask is already 100 x 100. But make it 200 x 200
		# mask-out self-contrast cases -
		# 가만 보니 this is just to prevent self dot product
		logits_mask = torch.scatter(
			torch.ones_like(mask),
			1, #dim is 1
			torch.arange(batch_size * anchor_count).view(-1, 1).to(device), #index
			0 #sorce
		)
		mask = mask * logits_mask  

		###################
		#Let's make mask for the same label
		#This is the mask for the denominator sum
		labels_concat = torch.cat([labels, labels], dim=0) #will have shape  200,1
		same_labels_mask = 1 - torch.eq(labels_concat, labels_concat.T).float().to(device)

		# compute log_prob
		if self.args.same_labels_mask:
			exp_logits = torch.exp(logits) * logits_mask * same_labels_mask  #everything itself myself dot myself
		else:
			exp_logits = torch.exp(logits) * logits_mask #everything itself myself dot myself
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

		# compute mean of log-likelihood over positive
		mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) #True인거만 곱하기 except for myself dot myself

		# loss
		loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
		loss = loss.view(anchor_count, batch_size).mean()

		return loss

	#Use objclr dataloader for train
	def train(self, train_loader, inference_train_datasets, test_loader, just_average=True, train_batch_size=1, eval_period = 1, train_sampler=None):
		scaler = GradScaler(enabled=self.args.fp16_precision)
		print("Start Training!")

		for epoch_counter in range(self.args.epochs):
			#Evaluate only at the 0th gpu
			# if epoch_counter  % eval_period ==0:# and epoch_counter  >0:
			# 	if self.args.local_rank ==0:
			# 		with torch.no_grad():
			# 			self.model.eval()
			# 			self.classify_inference(inference_train_datasets, test_loader, just_average, train_batch_size)

			# 	if self.args.multi_gpu:
			# 	

			if self.args.local_rank ==0:
				f = open(self.args.log_name, 'a')
				print("Epoch is ", epoch_counter)
				f.write("=========================================================="+ '\n')
				f.write("Epoch is "+ str(epoch_counter) + '\n')
			if self.args.multi_gpu:
				dist.barrier() 	

			self.model.train()
			if self.args.multi_gpu:
				train_sampler.set_epoch(epoch_counter)
			if epoch_counter  % eval_period ==0 and epoch_counter  >0:
				if self.args.local_rank ==0:
					with torch.no_grad():
						self.model.eval()
						self.classify_inference(inference_train_datasets, test_loader, f, just_average, train_batch_size)
						save_checkpoint(epoch_counter, self.model, self.args.model_name, '/projects/rsalakhugroup/soyeonm/objs_saved_models', multi_gpu = self.args.multi_gpu)
				if self.args.multi_gpu:
					dist.barrier() 
			
			mean_loss = 0.0
			batch_i = 0
			
			for batch_dict in tqdm(train_loader):
				#print("train_loader data loading time: ", time.time() - start)
				start = time.time()
				images_aug0 = torch.cat([batch_dict['image_' + str(i)][0] for i in range(self.args.object_views)])
				images_aug1 = torch.cat([batch_dict['image_' + str(i)][1] for i in range(self.args.object_views)])

				images = torch.cat([images_aug0,images_aug1] , dim=0)
				images = images.to(self.args.device, non_blocking=True)
				images_shape = images.shape
				#print("images shape is ", images.shape)

				if self.args.class_label:
					#Example: torch.cat([torch.arange(10).view(1, -1)]*5, dim=0).T.reshape(-1)
					#pickle.dump(batch_dict['category_label'], open("category_label.p", "wb"))
					#labels = torch.cat([batch_dict['category_label'].view(1,-1) for i in range(self.args.object_views)], dim=0).T.reshape(-1)
					labels = torch.cat([batch_dict['category_label'] for i in range(self.args.object_views)])
				else:
					#labels = torch.cat([batch_dict['object_label'].view(1,-1) for i in range(self.args.object_views)], dim=0).T.reshape(-1)
					labels = torch.cat([batch_dict['object_label'] for i in range(self.args.object_views)])
				#pickle.dump(labels, open("labels.p", "wb"))
				#labels = torch.cat([images_aug0,images_aug0] , dim=0) #labels should be for labels_aug0 only.
				labels = labels.to(self.args.device, non_blocking=True)

				with autocast(enabled=self.args.fp16_precision):
					features = self.model(images)
					del images; torch.cuda.empty_cache()
					#print("features shape is ", features.shape)
					features = F.normalize(features, dim=1)

					bsz = labels.shape[0] #Should be half the shape of images 

					#TODO: Check bsz is half of the shape of images
					assert bsz == images_shape[0]/2

					#Now follow the protocol of SupContrast
					f1, f2 = torch.split(features, [bsz, bsz], dim=0)
					features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #torch.Size([100, 2, 128])
					loss = self.sup_con_loss(features, labels)
					mean_loss += loss.detach().cpu().item()
					#print("loss is ", loss.detach().cpu().item())

				# SGD
				self.optimizer.zero_grad()

				scaler.scale(loss).backward()

				scaler.step(self.optimizer)
				scaler.update()
				batch_i +=1

			if epoch_counter >= 10:
				self.scheduler.step()

			if self.args.local_rank ==0:
				print("Epoch: " + str(epoch_counter) +"Mean Loss: " + str(mean_loss/ (batch_i+1)))
				print("Epoch: " + str(epoch_counter) +"Loss: " + str(loss))
				f.write("Epoch: " + str(epoch_counter) +"Mean Loss: " + str(mean_loss/ (batch_i+1)) + '\n')
				f.write("Epoch: " + str(epoch_counter) +"Loss: " + str(loss) + '\n')
				f.write("Time took: " + str(time.time() - start) + '\n')


			#Evaluate only at the 0th gpu
			if epoch_counter  % eval_period ==0 and epoch_counter  >0:
				if self.args.local_rank ==0:
					with torch.no_grad():
						self.model.eval()
						self.classify_inference(inference_train_datasets, test_loader, f, just_average, train_batch_size)
						save_checkpoint(epoch_counter, self.model, self.args.model_name, '/projects/rsalakhugroup/soyeonm/objs_saved_models', multi_gpu = self.args.multi_gpu)
				if self.args.multi_gpu:
					dist.barrier() 
			if self.args.local_rank ==0:
				f.close()

	#use permclr datasets for train_datasets, test_loader
	#trin_datasets have transform "None"
	def classify_inference(self, train_dataset, test_loader, f=None, just_average=True, train_batch_size=1):
		print("Start Inference!")
		class_alignment = []

		P_mat = get_perm_matrix_identity(self.args.object_views).to(self.args.device) #has shape 8x8 
		P_mat_128 = torch.cat([P_mat.unsqueeze(0)]*128, axis=0).float()

		avg_matrix = get_avg_matrix(self.args.object_views) #8x2
		avg_matrix_128 = torch.cat([avg_matrix.unsqueeze(0)]*128, axis=0).to(self.args.device)

		num_classes = len(train_dataset)
		#class_lens = [len(td) for td in train_dataset]

		#I THINK THIS ONLY WORKS FOR ONESHOT NOW (TRAIN)
		train_category_labels_tup =[]
		cat_by_category = []
		for ci in range(num_classes):
			batch_dict = train_dataset[ci]#[c] 
			cat_by_category += [batch_dict['image_' + str(i)].unsqueeze(0) for i in range(self.args.object_views)] #If we use dataloaders, we can do torch.Size([8, 3, 32, 32]) #8 is batch_size * num_objects (permclr_views) #Just remove unsqueeze(0) for dataloader
		catted_imgs = torch.cat(cat_by_category)
		train_category_labels_tup.append(catted_imgs)

		for batch_dict in tqdm(test_loader):
			test_images = torch.cat([batch_dict['image_' + str(i)] for i in range(self.args.object_views)]) #CHeck what this is exactly
			test_labels = batch_dict['category_label'].cpu().numpy()
			#test_labels = torch.cat([batch_dict['category_label'] for i in range(self.args.object_views)])
			#test_images = test_images.to(self.args.device, non_blocking=True)
			#test_labels = test_labels.to(self.args.device, non_blocking=True)

			batch_imgs = torch.cat(train_category_labels_tup + [test_images]).to(self.args.device, non_blocking=True)
			train_len_with_multi_views = batch_imgs.shape[0] - test_images.shape[0]
			test_len_with_multi_views = test_images.shape[0]

			train_len = int(train_len_with_multi_views/self.args.object_views); assert train_len * self.args.object_views == train_len_with_multi_views
			test_len = int(test_len_with_multi_views/self.args.object_views); assert test_len * self.args.object_views == test_len_with_multi_views


			with autocast(enabled=self.args.fp16_precision):
				features = self.model(batch_imgs)
				del test_images; torch.cuda.empty_cache()
				#Now separate into two
				features_train = features[:train_len_with_multi_views, :].clone() 
				#print("train shape ", features_train.shape)
				features_test = features[train_len_with_multi_views:, :].clone() 
				#print("test shape ", features_test.shape)
				del features; del batch_imgs

				#Stack and concatenate
				#Stack features_train first
				#assert int(train_len/self.args.object_views)*self.args.object_views == train_len
				features_train = features_train.reshape(train_len, self.args.object_views, -1)
				features_train = torch.cat([features_train]*test_len) #shape should be (len(test_images)*len(train object numbers), self.args.object_views, 128 )

				#SHOULD work from here
				#Stack features_test
				#assert int(features_test.shape[0]/self.args.object_views) * self.args.object_views == features_test.shape[0]
				#assert test_len == features_test.shape[0]
				ori_tet_shape = features_test.shape
				feature_test = features_test.reshape(test_len,self.args.object_views, -1).transpose(0,1).reshape(ori_tet_shape)
				features_test = features_test.reshape(test_len, self.args.object_views, -1) #ASSUME BATCH_SIZE=1
				assert features_test.shape[2] == self.args.out_dim
				features_test = features_test.transpose(0,1) #(self.args.object_views, num_classes, 128)
				features_test = torch.cat([features_test]*train_len) #(self.args.object_views*num_classes, num_classes, 128)
				features_test = features_test.transpose(0,1)
				features_test = features_test.reshape(test_len*train_len, self.args.object_views, -1) #shape is (num_classes**2*train_batch_size,, self.args.object_views, 128 ) WITH batch size 1

				#Do the same for test labels for later (for calculating accuracy)
				#test_labels = torch.cat([test_labels.unsqueeze(0)] *num_classes, dim=1) 
				#test_labels = test_labels.rehspae(-1) #Has shape len(test_labels) * num_classes

				#Concatenate feature_test and features_train 
				features = torch.cat([features_test, features_train], axis=1) #shape is (num_classes**2*train_batch_size,, self.args.object_views*2, 128 ) WITH batch size 1

				features = features.permute(2, 1, 0) #Now shape is 128 x self.args.object_views*2x num_classes**2 (used to be 36 x 8x 128)
				features = torch.bmm(P_mat_128, features) #shape is still 128, 8, 9 
				features = features.permute(0, 2, 1) #Shape is now 128 x 9 x 8. THIS IS (kind of? reshaped) THE PERMUTED B (B * P^T)

				#Get average
				features = torch.bmm(features, avg_matrix_128)

				#Take dot product
				#Normalize across 128
				assert features.shape[2] == 2
				features[:, :, 0] = F.normalize(features[:, :, 0].clone(), dim=0); features[:, :, 1] = F.normalize(features[:, :, 1].clone(), dim=0)
				#Multiply elementwise among the dimension of "2"
				features = torch.mul(features[:, :, 0].clone(), features[:,:,1].clone()) #Shape is torch.Size([128, 9]) or torch.Size([128, 9*17])
				#Now sum across the 128 dimensions
				logits = torch.sum(features, axis=0) #Shape is torch.Size([9*train_batch_size]) or torch.Size([9*train_batch_size*17])

				if (just_average):
					#train_len == num_classes*train_batch_size
					logits = logits.reshape(test_len*num_classes,train_batch_size) #The first tranin_batchsize are car_test, car_train(1,2,..,train_batch_size,), ...
					logits = torch.mean(logits, axis=1)

				logits= logits.detach().cpu().numpy()

				assert logits.shape[0] %num_classes ==0
				max_logits, aligns =  get_max_logit_refactored_march10(logits, test_labels, num_classes)
				class_alignment += aligns

		print("class alignment is ", np.mean(class_alignment))
		if not(f is None):
			f.write("class alignment is " + str( np.mean(class_alignment)) + '\n')


