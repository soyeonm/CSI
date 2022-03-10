import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from glob import glob
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
		return string.split('/')[1]


def get_simclr_pipeline_transform(size, s=1, resize_size=None):
  """Return a set of data augmentation transformations as described in the SimCLR paper."""
  color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
  transform_list = [transforms.RandomResizedCrop(size=size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        GaussianBlur(kernel_size=int(0.1 * size)),
                                        transforms.ToTensor()]
  if not(resize_size is None):
    transform_list = [transforms.Resize((resize_size, resize_size))]+ transform_list
                                          
  data_transforms = transforms.Compose(transform_list)
  return data_transforms

class ObjDataset(Dataset):
	def __init__(self, root_dir, views, transform=None, ood_classes=None, processed=True):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""

		#self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir #e.g. /home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj
		#self.class2idx = {'hairdryer':0, 'suitcase':1, 'broccoli': 2}
		caegory_globs = glob(os.path.join(self.root_dir, '*'))
		if not (ood_classes is None):
			ood_classes_set = set(ood_classes)
			caegory_globs = [c for c in caegory_globs if not(c in ood_classes_set)]
		
		self.class2idx = {c.split('/')[-1]: i for i, c in enumerate(caegory_globs)}
		#print("class2idx is ", self.class2idx)
		globs = []
		for c in caegory_globs:
			if processed:
				globs += glob(c + '/*.jpg') #For all the rest
			else:
				globs += glob(c + '/*/*/*.jpg') #For '/projects/rsalakhugroup/soyeonm/co3d/co3d_march_9_classify/train'
		if processed:
			jpgs = [g.split('/')[-1] for g in globs]
		else:
			jpgs = copy.deepcopy(globs)
		#class_labels = [g.split('/')[-2] for g in globs]
		#This is taking so much time
		#self.object_dict = {get_obj_num(jpg): glob(os.path.join(self.root_dir, category, 'obj' + get_obj_num(jpg) +'*')) for jpg in set(jpgs)}
		object_ids = set([get_obj_num(jpg, processed) for jpg in set(jpgs)])
		self.object_dict_p = {o:[] for o in object_ids}
		self.object_class_dict_p = {o:None for o in object_ids}
		for g in globs:
			if processed:
				jpg = g.split('/')[-1]
			else:
				jpg = g
			if processed:
				class_label = g.split('/')[-2]
			else:
				class_label = g.split('/')[-4]
			obj_id = get_obj_num(jpg, processed) #used to be just get_obj_num(jpg) but adding classlabel to prevent overlap just in cases
			#if not(obj_id in object_ids):
			self.object_dict_p[obj_id].append(g)
			self.object_class_dict_p[obj_id] = class_label


		self.object_dict = {i: self.object_dict_p[k] for i, k in enumerate(list(self.object_dict_p.keys()))} #object id to jpg paths
		self.object_class_dict = {i: self.object_class_dict_p[k] for i, k in enumerate(list(self.object_dict_p.keys()))}
		self.views = views
		self.transform = transform
		#self.resize_transform = transforms.Resize((resize_shape, resize_shape))
		self.t = transforms.ToTensor()
		pickle.dump(self.object_dict, open("temp_pickles/object_dict.p", "wb"))
		pickle.dump(self.object_class_dict, open("temp_pickles/object_class_dict.p", "wb"))
		del self.object_dict_p; del self.object_class_dict_p

		#get the number of unique classes
		#self.class_lens = len(set(list(self.object_class_dict.values())))

	def shuffle(self, seed):
		#shuffle key
		new_object_dict = {}
		np.random.seed(seed)
		permute = np.random.permutation(len(self.object_dict)).tolist()
		for i, k in enumerate(list(self.object_dict.keys())):
			new_object_dict[permute[i]] = self.object_dict[k]
		self.object_dict = new_object_dict
		del new_object_dict

		#shuffle value
		k_count = 1
		for k in self.object_dict:
			np.random.seed(seed + k_count)
			np.random.shuffle(self.object_dict[k])
			k_count +=1

	def __len__(self):
		return len(self.object_dict)

	#Should return the object in the "idx"
	#Sample "view" views from the object
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

			#image = self.resize_transform(image)
			if self.transform is not None:
				image = self.transform(image)

			if self.transform is None:
				image = self.t(image)

			return_dict['image_' + str(v)] = image
		return_dict['object_label'] = idx
		return_dict['category_label'] = self.class2idx[self.object_class_dict[idx]]

		return return_dict

# class ObjSupClrDataset(Dataset):
# 	def __init__(self, root_dir, views, resize_shape, transform=None):
# 		"""
# 		Args:
# 			csv_file (string): Path to the csv file with annotations.
# 			root_dir (string): Directory with all the images.
# 			transform (callable, optional): Optional transform to be applied
# 				on a sample.
# 		"""

# 		self.root_dir = root_dir #e.g. /home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj
# 		caegory_globs = glob(os.path.join(self.root_dir, '*'))
# 		globs = []
# 		for c in caegory_globs:
# 			globs += glob(c + '/*')
# 		jpgs = [g.split('/')[-1] for g in globs]

# 		self.jpg_object_labels = {}
# 		self.jpg_class_labels = {}
# 		for gi, g in enumerate(globs):
# 			jpg = g.split('/')[-1]
# 			class_label = g.split('/')[-2]
# 			obj_id = get_obj_num(jpg)
# 			self.jpg_object_labels[jpg] = obj_id

# 			#if not(obj_id in object_ids):
# 			self.object_dict_p[obj_id].append(g)
# 			self.object_class_dict_p[obj_id] = class_label


# 		self.object_dict = {i: self.object_dict_p[k] for i, k in enumerate(list(self.object_dict.keys()))} #object id to jpg paths
# 		self.object_class_dict = {i: self.object_class_dict_p[k] for i, k in enumerate(list(self.object_dict.keys()))}
# 		self.views = views
# 		self.transform = transform
# 		self.resize_transform = transforms.Resize((resize_shape, resize_shape))
# 		self.t = transforms.ToTensor()
# 		del self.object_dict_p; del self.object_class_dict

# 	def shuffle(self, seed):
# 		#shuffle key
# 		new_object_dict = {}
# 		np.random.seed(seed)
# 		permute = np.random.permutation(len(self.object_dict)).tolist()
# 		for i, k in enumerate(list(self.object_dict.keys())):
# 			new_object_dict[permute[i]] = self.object_dict[k]
# 		self.object_dict = new_object_dict
# 		del new_object_dict

# 		#shuffle value
# 		k_count = 1
# 		for k in self.object_dict:
# 			np.random.seed(seed + k_count)
# 			np.random.shuffle(self.object_dict[k])
# 			k_count +=1

# 	def __len__(self):
# 		return len(self.object_dict)

# 	#Should return the object in the "idx"
# 	#Sample "view" views from the object
# 	#Just return image, label
# 	def __getitem__(self, idx):
# 		#Get object from idx
# 		return_dict = {}
# 		object_paths = self.object_dict[idx]
# 		#Sample 4(self.views) images at random from here
# 		sample_view_indices = np.random.permutation(len(object_paths))[:self.views]
# 		#Open and concatenate
# 		for v in range(self.views):
# 			im_path = object_paths[sample_view_indices[v]]
# 			image = default_loader(im_path)

# 			image = self.resize_transform(image)
# 			if self.transform is not None:
# 				image = self.transform(image)

# 			if self.transform is None:
# 				image = self.t(image)

# 			return_dict['image_' + str(v)] = image
# 		return_dict['object_label'] = idx
# 		return_dict['category_label'] = self.object_class_dict[idx]

# 		return return_dict

