#Maybe do like Permclr but use SupContrast loss

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
from permclr import get_perm_matrix_identity, get_avg_matrix, get_max_logit

torch.manual_seed(0)


class ObjCLR(object):
	def __init__(self, temperature=0.07, contrast_mode='all',
				 base_temperature=0.07, *args, **kwargs):
		self.args = kwargs['args']
		self.model = kwargs['model'].to(self.args.device)
		self.optimizer = kwargs['optimizer']
		self.scheduler = kwargs['scheduler']
		self.writer = SummaryWriter()
		logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
		self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)


		self.temperature = temperature
		self.contrast_mode = contrast_mode
		self.base_temperature = base_temperature

	def sup_con_loss(self, features, labels=None, mask=None):
		if len(features.shape) < 3:
			raise ValueError('`features` needs to be [bsz, n_views, ...],'
							 'at least 3 dimensions are required')
		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		batch_size = features.shape[0]
		if labels is not None and mask is not None:
			raise ValueError('Cannot define both `labels` and `mask`')
		elif labels is None and mask is None:
			mask = torch.eye(batch_size, dtype=torch.float32).to(self.args.device)
		elif labels is not None:
			labels = labels.contiguous().view(-1, 1)
			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')
			mask = torch.eq(labels, labels.T).float().to(self.args.device)
		else:
			mask = mask.float().to(self.args.device)

		contrast_count = features.shape[1]
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
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
			torch.matmul(anchor_feature, contrast_feature.T),
			self.temperature)
		# for numerical stability
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()

		# tile mask
		mask = mask.repeat(anchor_count, contrast_count)
		# mask-out self-contrast cases
		logits_mask = torch.scatter(
			torch.ones_like(mask),
			1,
			torch.arange(batch_size * anchor_count).view(-1, 1).to(self.args.device),
			0
		)
		mask = mask * logits_mask

		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

		# compute mean of log-likelihood over positive
		mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

		# loss
		loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
		loss = loss.view(anchor_count, batch_size).mean()

		return loss

	#Use objclr dataloader for train
	def train(self, train_loader, inference_train_datasets, test_loaders, class_lens, just_average=True, train_batch_size=1, eval_period = 1):
		scaler = GradScaler(enabled=self.args.fp16_precision)
		print("Start Training!")

		for epoch_counter in range(self.args.epochs):
			self.model.train()
			print("Epoch is ", epoch_counter)
			mean_loss = 0.0
			batch_i = 0
			for batch_dict in tqdm(train_loader):
				images_aug0 = torch.cat([batch_dict['image_' + str(i)][0] for i in range(self.args.object_views)])
				images_aug1 = torch.cat([batch_dict['image_' + str(i)][1] for i in range(self.args.object_views)])

				images = torch.cat([images_aug0,images_aug1] , dim=0)
				images = images.to(self.args.device, non_blocking=True)

				if self.args.class_label:
					labels = torch.cat([batch_dict['category_label'] for i in range(self.args.object_views)])
				else:
					labels = torch.cat([batch_dict['object_label'] for i in range(self.args.object_views)])
				#labels = torch.cat([images_aug0,images_aug0] , dim=0) #labels should be for labels_aug0 only.
				labels = labels.to(self.args.device, non_blocking=True)

				with autocast(enabled=self.args.fp16_precision):
					features = self.model(images)
					features = F.normalize(features, dim=1)

					bsz = labels.shape[0] #Should be half the shape of images 

					#TODO: Check bsz is half of the shape of images
					assert bsz == images.shape[0]/2

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

			print("Epoch: " + str(epoch_counter) +"Mean Loss: " + str(mean_loss/ (batch_i+1)))
			print("Epoch: " + str(epoch_counter) +"Loss: " + str(loss))

			if epoch_counter  % eval_period ==0:# and epoch_counter  >0:
				with torch.no_grad():
					self.model.eval()
					self.classify_inference(inference_train_datasets, test_loaders, class_lens, just_average, train_batch_size)

	#use permclr datasets for train_datasets, test_loader
	#trin_datasets have transform "None"
	def classify_inference(self, train_datasets, test_loaders, class_lens , just_average=True, train_batch_size=1):
		print("Start Inference!")
		class_alignment = []

		P_mat = get_perm_matrix_identity(self.args.object_views).to(self.args.device) #has shape 8x8 
		P_mat_128 = torch.cat([P_mat.unsqueeze(0)]*128, axis=0).float()

		avg_matrix = get_avg_matrix(self.args.object_views) #8x2
		avg_matrix_128 = torch.cat([avg_matrix.unsqueeze(0)]*128, axis=0).to(self.args.device)

		num_classes = len(train_datasets)
		class_lens = [len(td) for td in train_datasets]
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
				cat_by_category += [batch_dict['image_' + str(i)].unsqueeze(0) for i in range(self.args.object_views)]
			catted_imgs = torch.cat(cat_by_category)
			train_category_labels_tup.append(catted_imgs)

		for batch_i, batch_dict_tuple in enumerate(itertools.zip_longest(*test_loaders)): 
			none_mask = []
			#catted_img_tups of test dataset
			catted_imgs_tup = []
			object_labels_tup = []
			category_labels_tup =[]
			#concatente all the image_i's together in one direction(image_0: all the image_0's, image_3's: all the image_3's)
			for batch_dict in batch_dict_tuple:
				if not(batch_dict is None):
					catted_imgs = torch.cat([batch_dict['image_' + str(i)] for i in range(self.args.object_views)]) #shape is torch.Size([8, 3, 32, 32]) #8 is batch_size * num_objects (permclr_views)
					print("catted_imgs shape ", catted_imgs.shape)
					if not(self.args.ood):
						object_labels = torch.cat([batch_dict['object_label'] for i in range(self.args.object_views)]) #shape is torch.Size([8])
						category = self.args.classes_to_idx[batch_dict['category_label'][0]]
						category_labels_tup.append(torch.tensor([category]*self.args.object_views*self.args.batch_size))
						object_labels_tup.append(object_labels)
					none_mask.append(False)
				else:
					#pass
					none_mask.append(True)
				catted_imgs_tup.append(catted_imgs)

			batch_imgs = torch.cat(train_category_labels_tup + catted_imgs_tup)
			batch_imgs = batch_imgs.to(self.args.device)

			with autocast(enabled=self.args.fp16_precision):
				features = self.model(batch_imgs)

				#Now separate into two
				features_train = features[:self.args.object_views*num_classes*train_batch_size, :].clone() 
				print("train shape ", features_train.shape)
				features_test = features[self.args.object_views*num_classes*train_batch_size:, :].clone() 
				print("test shape ", features_test.shape)
				del features

				#Stack and concatenate
				#Stack features_train first
				features_train = features_train.reshape(num_classes*train_batch_size, self.args.object_views, -1)
				features_train = torch.cat([features_train]*num_classes) #ASSUME BATCH_SIZE=1 #CHANGE FROM HERE IF CHANGE BATCH SIZE #shape should be (num_classes**2*train_batch_size, self.args.object_views, 128 )

				#Stack features_test
				features_test = features_test.reshape(num_classes, self.args.object_views, -1) #ASSUME BATCH_SIZE=1
				features_test = features_test.transpose(0,1) #(self.args.object_views, num_classes, 128)
				features_test = torch.cat([features_test]*num_classes*train_batch_size) #(self.args.object_views*num_classes, num_classes, 128)
				features_test = features_test.transpose(0,1)
				features_test = features_test.reshape(num_classes**2*train_batch_size, self.args.object_views, -1) #shape is (num_classes**2*train_batch_size,, self.args.object_views, 128 ) WITH batch size 1

				#Concatenate feature_test and features_train 
				features = torch.cat([features_test, features_train], axis=1) #shape is (num_classes**2*train_batch_size,, self.args.object_views*2, 128 ) WITH batch size 1

				features = features.permute(2, 1, 0) #Now shape is 128 x self.args.object_views*2x num_classes**2 (used to be 36 x 8x 128)
				features = torch.bmm(P_mat_128, features) #shape is still 128, 8, 9 
				features = features.permute(0, 2, 1) #Shape is now 128 x 9 x 8. THIS IS (kind of? reshaped) THE PERMUTED B (B * P^T)

				#Get average
				features = torch.bmm(features, avg_matrix_128)

				#Take dot product
				#Normalize across 128
				features[:, :, 0] = F.normalize(features[:, :, 0].clone(), dim=0); features[:, :, 1] = F.normalize(features[:, :, 1].clone(), dim=0)
				#Multiply elementwise among the dimension of "2"
				features = torch.mul(features[:, :, 0].clone(), features[:,:,1].clone()) #Shape is torch.Size([128, 9]) or torch.Size([128, 9*17])
				#Now sum across the 128 dimensions
				logits = torch.sum(features, axis=0) #Shape is torch.Size([9*train_batch_size]) or torch.Size([9*train_batch_size*17])

				if (just_average):
					logits = logits.reshape(num_classes**2,train_batch_size) #The first tranin_batchsize are car_test, car_train(1,2,..,train_batch_size,), ...
					logits = torch.mean(logits, axis=1)

				logits= logits.detach().cpu().numpy()
				for ni, m in enumerate(none_mask):
					if m == True:
						logits[3*ni:3*(ni+1)]= np.nan

				assert logits.shape[0] %3 ==0
				max_logits, aligns =  get_max_logit(logits, none_mask)
				class_alignment += aligns

		print("class alignment is ", np.mean(class_alignment))


