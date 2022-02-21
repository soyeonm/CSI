import time

import torch.optim

import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, NT_xent
from utils.utils import AverageMeter, normalize
import pickle

import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from SimCLR.utils import save_config_file, accuracy, save_checkpoint

import pickle



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        ### SimCLR loss ###
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(device)
            images_pair = hflip(images.repeat(2, 1, 1, 1))  # 2B with hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(device), images[1].to(device)
            images_pair = torch.cat([images1, images2], dim=0)  # 2B

        pickle.dump(images.detach().cpu(), open("images.p", "wb"))
        pickle.dump(images_pair.detach().cpu(), open("images_pair.p", "wb"))
        labels = labels.to(device)
        if P.print_batch_size:
            print("batch size is ", batch_size)

        images_pair = simclr_aug(images_pair)  # transform
        pickle.dump(images_pair.detach().cpu(), open("aug_images_pair.p", "wb"))

        _, outputs_aux = model(images_pair, simclr=True, penultimate=True)
        pickle.dump(outputs_aux, open("outputs_aux.p", "wb"))

        simclr = normalize(outputs_aux['simclr'])  # normalize
        pickle.dump(simclr.detach().cpu(), open("simclr.p", "wb"))
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
        pickle.dump(sim_matrix.detach().cpu(), open("sim_matrix.p", "wb"))
        loss_sim = NT_xent(sim_matrix, temperature=0.5) * P.sim_lambda
        pickle.dump(loss_sim.detach().cpu(), open("loss_sim.p", "wb"))

        ### total loss ###
        loss = loss_sim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Post-processing stuffs ###
        simclr_norm = outputs_aux['simclr'].norm(dim=1).mean()

        ### Linear evaluation ###
        outputs_linear_eval = linear(outputs_aux['penultimate'].detach())
        loss_linear = criterion(outputs_linear_eval, labels.repeat(2))

        linear_optim.zero_grad()
        loss_linear.backward()
        linear_optim.step()

        ### Log losses ###
        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossSim %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value))

        check = time.time()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['sim'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
