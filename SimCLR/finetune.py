import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import argparse

from torchvision import models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--load_path', type=str, required=True)
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-dataset-name', default='co3d',
                    help='dataset name', choices=['stl10', 'cifar10', 'co3d'])
parser.add_argument('--dn', type=str, required=True)

args = parser.parse_args()


def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.CIFAR10('./data', train=True, download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=10, drop_last=False, shuffle=True)
  
  test_dataset = datasets.CIFAR10('./data', train=False, download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=False)
  return train_loader, test_loader


def get_co3d_data_loaders(batch_size=256):
  train_dataset = datasets.ImageFolder('../data/co3d_small_split_one_no_by_obj/train', transform=transforms.ToTensor())#Maybe add resize too
  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=10, drop_last=False, shuffle=True)
  
  test_dataset = datasets.ImageFolder('../data/co3d_small_split_one_no_by_obj/test', transform=transforms.ToTensor())#Maybe add resize too

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=False)
  return train_loader, test_loader


if args.arch == 'resnet18':
  model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
elif args.arch == 'resnet50':
  model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)


#Load from path

checkpoint = torch.load(args.load_path, map_location=device)
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):

  if k.startswith('backbone.'):
    if k.startswith('backbone') and not k.startswith('backbone.fc'):
      # remove prefix
      state_dict[k[len("backbone."):]] = state_dict[k]
  del state_dict[k]

log = model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']

#Now load data
train_loader, test_loader = get_co3d_data_loaders()

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(device)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

#Train and save
epochs = 100
best_accuracy = -100
best_state_dict = None
for epoch in range(epochs):
  top1_train_accuracy = 0
  for counter, (x_batch, y_batch) in enumerate(train_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    
    top1 = accuracy(logits, y_batch, topk=(1,))
    top1_train_accuracy += top1[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  top1_train_accuracy /= (counter + 1)
  top1_accuracy = 0
  top5_accuracy = 0
  for counter, (x_batch, y_batch) in enumerate(test_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
  
    top1, top5 = accuracy(logits, y_batch, topk=(1,5))
    top1_accuracy += top1[0]
    top5_accuracy += top5[0]
  
  top1_accuracy /= (counter + 1)
  top5_accuracy /= (counter + 1)
  best_accuracy = max(best_accuracy, top1_accuracy)
  print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
  print("Best accuracy so far is ", best_accuracy)
  if best_accuracy == top1_accuracy:
  	best_state_dict = self.model.state_dict()
print("Finisehd training and saving the best!")

save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': best_state_dict,
            'optimizer': self.optimizer.state_dict(),
        }, is_best=True, filename='finetuned_models' + args.dn + '.pth')



