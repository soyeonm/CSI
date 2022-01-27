from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from glob import glob

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)


def default_loader(path: str) -> Any:
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)


def get_obj_num(string):
	#between obj and frame
	assert string[:3] == 'obj'
	i = string.find('frame')
	#and string[20:25] == 'frame'
	return string[3:i]

class PermDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, root_dir, category, views, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""

		#self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir #e.g. /home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj
		self.category = category
		globs = glob(os.path.join(self.root_dir, category, '*'))
		jpgs = [g.split('/')[-1] for g in globs]
		self.object_dict = {get_obj_num(jpg): glob(os.path.join(self.root_dir, category, 'obj' + get_obj_num(jpg) +'*')) for jpg in set(jpgs)}
		self.object_dict = {i: self.object_dict[k] for i, k in enumerate(list(self.object_dict.keys()))}
		self.views = views
		self.transform = transform

	#Shuffle the value lists of self.object_dict
	def shuffle(self):
		for k in self.object_dict:
			np.random.shuffle(self.object_dict[k])

	def __len__(self):
		return len(self.object_dict)

	#Should return the object in the "idx"
	#Batch size will be 2 
	#Sample 4 views from the object
	#Just return image, label
	def __getitem__(self, idx):
		#Get object from idx
		return_dict = {}
		object_paths = self.object_dict[idx]
		#Sample 4(self.views) images at random from here
		sample_view_indices = np.random.permutation(len(object_paths))[:self.views]
		#Open and concatenate
		for v in range(self.views):
			im_path = object_paths[sample_view_indices[v]]
			image = default_loader(im_path)

			if self.transform is not None:
				image = self.transform(image)

			return_dict['image_' + str(v)] = image
		return_dict['object_label'] = idx
		return_dict['category_label'] = category

		return return_dict

class PermMultipleDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, root_dir, classes, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.landmarks_frame)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = os.path.join(self.root_dir,
								self.landmarks_frame.iloc[idx, 0])
		image = io.imread(img_name)
		landmarks = self.landmarks_frame.iloc[idx, 1:]
		landmarks = np.array([landmarks])
		landmarks = landmarks.astype('float').reshape(-1, 2)
		sample = {'image': image, 'landmarks': landmarks}

		if self.transform:
			sample = self.transform(sample)

		return sample