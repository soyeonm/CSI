import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from glob import glob
import pickle
import copy

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path):
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)


def default_loader(path):
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)


def get_obj_num(string, processed):
	if processed:
		assert string[:3] == 'obj', "string is " + str(string)
		i = string.find('frame')
		#and string[20:25] == 'frame'
		return string[3:i]
	else:
		return string.split('/')[-3]

class PermDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, root_dir, category, views, resize_shape, transform=None, processed=True):
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
		if processed:
			globs = glob(os.path.join(self.root_dir, category, '*.jpg'))
		else:
			globs = glob(os.path.join(self.root_dir, category, '*/*/*.jpg'))
		pickle.dump(globs, open("temp_pickles/perm_globs.p", "wb"))
		if processed:
			jpgs = [g.split('/')[-1] for g in globs]
		else:
			jpgs = copy.deepcopy(globs)
		if not(processed):
			pickle.dump(jpgs, open("temp_pickles/perm_jpgs.p", "wb"))
		#This is taking so much time
		#self.object_dict = {get_obj_num(jpg): glob(os.path.join(self.root_dir, category, 'obj' + get_obj_num(jpg) +'*')) for jpg in set(jpgs)}
		object_ids = set([get_obj_num(jpg, processed) for jpg in set(jpgs)])
		if not(processed):
			pickle.dump(jpgs, open("temp_pickles/objectids.p", "wb"))
		self.object_dict = {o:[] for o in object_ids}
		for g in globs:
			if processed:
				jpg = g.split('/')[-1]
			else:
				jpg = g
			obj_id = get_obj_num(jpg, processed)
			#if not(obj_id in object_ids):
			self.object_dict[obj_id].append(g)
		

		self.object_dict = {i: self.object_dict[k] for i, k in enumerate(list(self.object_dict.keys()))}
		if not(processed):
			pickle.dump(self.object_dict, open("temp_pickles/object_dict.p", "wb"))
		self.views = views
		self.transform = transform
		self.resize_transform = transforms.Resize((resize_shape, resize_shape))
		self.t = transforms.ToTensor()

	#Shuffle the key and value lists of self.object_dict
	def shuffle(self):
		#shuffle key
		new_object_dict = {}
		permute = np.random.permutation(len(self.object_dict)).tolist()
		for i, k in enumerate(list(self.object_dict.keys())):
			new_object_dict[permute[i]] = self.object_dict[k]
		self.object_dict = new_object_dict
		del new_object_dict

		#shuffle value
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

			image = self.resize_transform(image)
			if self.transform is not None:
				image = self.transform(image)

			if self.transform is None:
				image = self.t(image)

			return_dict['image_' + str(v)] = image
		return_dict['object_label'] = idx
		return_dict['category_label'] = self.category

		return return_dict

#Should not be needed
# class PermMultipleDataset(Dataset):
# 	"""Face Landmarks dataset."""

# 	def __init__(self, root_dir, classes, transform=None):
# 		self.root_dir = root_dir
# 		self.transform = transform
# 		self.classes = classes


# 	def __len__(self):
# 		return len(self.landmarks_frame)

# 	def __getitem__(self, idx):
# 		if torch.is_tensor(idx):
# 			idx = idx.tolist()

# 		img_name = os.path.join(self.root_dir,
# 								self.landmarks_frame.iloc[idx, 0])
# 		image = io.imread(img_name)
# 		landmarks = self.landmarks_frame.iloc[idx, 1:]
# 		landmarks = np.array([landmarks])
# 		landmarks = landmarks.astype('float').reshape(-1, 2)
# 		sample = {'image': image, 'landmarks': landmarks}

# 		if self.transform:
# 			sample = self.transform(sample)

# 		return sample