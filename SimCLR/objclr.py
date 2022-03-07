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
	def train(self, train_loader):
		scaler = GradScaler(enabled=self.args.fp16_precision)
		print("Start Training!")

		for epoch_counter in range(self.args.epochs):
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
					bsz = labels.shape[0] #Should be half the shape of images 

					#TODO: Check bsz is half of the shape of images
					assert bsz == images.shape[0]/2

					#Now follow the protocol of SupContrast
					f1, f2 = torch.split(features, [bsz, bsz], dim=0)
					features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #torch.Size([100, 2, 128])
					loss = self.sup_con_loss(features, labels)
					mean_loss += loss.detach().cpu().item()
					print("loss is ", loss.detach().cpu().item())

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


	#Use permclr dataloader for inference
	def inference(self):
		pass