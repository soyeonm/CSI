import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from glob import glob
import copy
import pickle
import cv2


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


def get_simclr_pipeline_transform(size, s=1, resize_size=None, crop_from=0.08):
  """Return a set of data augmentation transformations as described in the SimCLR paper."""
  color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
  transform_list = [transforms.RandomResizedCrop(size=size, scale=(crop_from, 1.0)),
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
	def __init__(self, root_dir, views, resize_shape= 32,transform=None, ood_classes=None, processed=True, mask=False):
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
		self.resize_transform = transforms.Resize((resize_shape, resize_shape))
		self.t = transforms.ToTensor()
		del self.object_dict_p; del self.object_class_dict_p
		self.mask = mask

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
			if self.mask:
				mask_path = im_path.replace('images', 'masks').replace('jpg', 'png').replace(self.root_dir, '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_14_masks')
				mask = cv2.imread(mask_path)
				mask = cv2.resize(mask, (300,300))
				wheres = np.where(mask !=0)
				start_crop = wheres[0][0]
				end_crop = wheres[1][0]
				image = np.asarray(image)[start_crop[0]:start_crop[1], end_crop[0]:end_crop[1]]
				image = Image.fromarray(np.uint8(image))
			
			if self.transform is not None:
				image = self.transform(image)

			if self.transform is None:
				image = self.resize_transform(image)
				image = self.t(image)

			return_dict['image_' + str(v)] = image
		return_dict['object_label'] = idx
		return_dict['category_label'] = self.class2idx[self.object_class_dict[idx]]

		return return_dict

#Essentially PermclrDataset but picking objects from each class, not just any object of random class
#Keep an index of object number per class, and the ordering of classes
#Actually no 
#Just access by class_number
class ObjInferenceDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, root_dir, views, resize_shape=32, shots=None, transform=None, ood_classes=None, processed=True, class_idx = None, sample=None, mask=False):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		assert (sample is None) or (shots is None)
		#self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir #e.g. /home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj
		#self.class2idx = {'hairdryer':0, 'suitcase':1, 'broccoli': 2}
		caegory_globs = glob(os.path.join(self.root_dir, '*'))
		if not (ood_classes is None):
			ood_classes_set = set(ood_classes)
			caegory_globs = [c for c in caegory_globs if not(c in ood_classes_set)]
		
		if class_idx is None: 
			self.class2idx = {c.split('/')[-1]: i for i, c in enumerate(caegory_globs)}
		else:
			self.class2idx = class_idx
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
		#self.object_class_dict_p = {o:None for o in object_ids}
		self.object_class_dict_p_rev = {c: [] for c in self.class2idx}
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
			#self.object_class_dict_p[obj_id] = class_label
			self.object_class_dict_p_rev[class_label].append(obj_id)

		self.object_class_dict_p_rev = {c: list(set(v)) for c, v in self.object_class_dict_p_rev.items()}

		################Do here differently from ObjDataset
		self.object_dict = {}
		self.object_class_dict = {}
		self.class2_startidx = {c: None for c in self.class2idx}

		#Sample shots for each class
		if not(shots is None):
			chosen_shots = []
			seed_counter = 0
			for c, v in self.object_class_dict_p_rev.items():
				np.random.seed(seed_counter)
				perm = np.random.permutation(len(v)).tolist()
				perm = perm[:shots]
				self.object_class_dict_p_rev[c] = [v[p] for p in perm]
				seed_counter +=1

		if not(sample is None):
			chosen_shots = []
			seed_counter = 0
			for c, v in self.object_class_dict_p_rev.items():
				np.random.seed(seed_counter)
				perm = np.random.permutation(len(v)).tolist()
				perm = perm[:int(len(v)*sample)]
				self.object_class_dict_p_rev[c] = [v[p] for p in perm]
				seed_counter +=1

		o_counter = 0
		for c in self.object_class_dict_p_rev:
			self.class2_startidx[c] = o_counter
			for o in self.object_class_dict_p_rev[c]:
				self.object_dict[o_counter] = o
				self.object_class_dict[o_counter] = c
				o_counter +=1 

		
		self.object_dict = {i: self.object_dict_p[v] for i, v in self.object_dict.items()} 
		#self.object_class_dict = {i: self.object_class_dict_p[k] for i, k in enumerate(list(self.object_dict_p.keys()))}
		#self.object_dict = {i: self.object_dict_p[k] for i, k in enumerate(list(self.object_dict_p.keys()))} #object id to jpg paths
		pickle.dump(self.object_class_dict, open("temp_pickles/new_object_class_dict.p", "wb"))
		pickle.dump(self.object_dict, open("temp_pickles/new_object_dict.p", "wb"))
		
		self.views = views
		self.transform = transform
		self.resize_transform = transforms.Resize((resize_shape, resize_shape))
		self.t = transforms.ToTensor()
		del self.object_dict_p#; del self.object_class_dict_p
		self.mask = mask

		#get the number of unique classes
		#self.class_lens = len(set(list(self.object_class_dict.values())))


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
			if self.mask:
				if self.root_dir == '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_9_classify/train':
					mask_path = im_path.replace('images', 'masks').replace('jpg', 'png')
				elif self.root_dir == '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_9_classify_real/test':
					last_jpg = im_path.split('/')[-1]; rest = '/'.join(im_path.split('/')[:-1])
					last_jpg = last_jpg[4:].replace('_f', '/masks/f').replace('jpg', 'png')
					mask_path = os.path.join(im_path, last_jpg)
				else:
					raise Exception("root dir invalid")
				mask_path = mask_path.replace(self.root_dir, '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_14_masks')
				mask = cv2.imread(mask_path)
				mask = cv2.resize(mask, (300,300))
				wheres = np.where(mask !=0)
				start_crop = wheres[0][0]
				end_crop = wheres[1][0]
				image = np.asarray(image)[start_crop[0]:start_crop[1], end_crop[0]:end_crop[1]]
				image = Image.fromarray(np.uint8(image))

			#image = self.resize_transform(image)
			if self.transform is not None:
				image = self.transform(image)

			if self.transform is None:
				image = self.resize_transform(image)
				image = self.t(image)

			return_dict['image_' + str(v)] = image
		return_dict['object_label'] = idx
		return_dict['category_label'] = self.class2idx[self.object_class_dict[idx]]

		return return_dict

####################################################################
#Added march 11 for efficiency
#THIS DATASET JUST RETURNS each img instance
#It keeps start index/ end index of each object
#and also start index/ end index of each 
# class ObjImgInstanceDataset(Dataset):
# 	def __init__(self, root_dir, views, resize_shape= 32,transform=None, ood_classes=None, processed=True):
# 		"""
# 		Args:
# 			csv_file (string): Path to the csv file with annotations.
# 			root_dir (string): Directory with all the images.
# 			transform (callable, optional): Optional transform to be applied
# 				on a sample.
# 		"""

# 		#self.landmarks_frame = pd.read_csv(csv_file)
# 		self.root_dir = root_dir #e.g. /home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_small_split_one_no_by_obj
# 		#self.class2idx = {'hairdryer':0, 'suitcase':1, 'broccoli': 2}
# 		caegory_globs = glob(os.path.join(self.root_dir, '*'))
# 		if not (ood_classes is None):
# 			ood_classes_set = set(ood_classes)
# 			caegory_globs = [c for c in caegory_globs if not(c in ood_classes_set)]
		
# 		self.class2idx = {c.split('/')[-1]: i for i, c in enumerate(caegory_globs)}
# 		#print("class2idx is ", self.class2idx)
# 		globs = []
# 		for c in caegory_globs:
# 			if processed:
# 				globs += glob(c + '/*.jpg') #For all the rest
# 			else:
# 				globs += glob(c + '/*/*/*.jpg') #For '/projects/rsalakhugroup/soyeonm/co3d/co3d_march_9_classify/train'
# 		if processed:
# 			jpgs = [g.split('/')[-1] for g in globs]
# 		else:
# 			jpgs = copy.deepcopy(globs)
# 		#class_labels = [g.split('/')[-2] for g in globs]
# 		#This is taking so much time
# 		#self.object_dict = {get_obj_num(jpg): glob(os.path.join(self.root_dir, category, 'obj' + get_obj_num(jpg) +'*')) for jpg in set(jpgs)}
# 		object_ids = set([get_obj_num(jpg, processed) for jpg in set(jpgs)])
# 		self.object_dict_p = {o:[] for o in object_ids}
# 		self.object_class_dict_p = {o:None for o in object_ids}
# 		self.object_class_dict_p_rev = {c:[] for c in self.class2idx}
# 		for g in globs:
# 			if processed:
# 				jpg = g.split('/')[-1]
# 			else:
# 				jpg = g
# 			if processed:
# 				class_label = g.split('/')[-2]
# 			else:
# 				class_label = g.split('/')[-4]
# 			obj_id = get_obj_num(jpg, processed) #used to be just get_obj_num(jpg) but adding classlabel to prevent overlap just in cases
# 			#if not(obj_id in object_ids):
# 			self.object_dict_p[obj_id].append(g)
# 			self.object_class_dict_p[obj_id] = class_label
# 			self.object_class_dict_p_rev[class_label].append(obj_id)

# 		self.object_class_dict_p_rev = {c: list(set(v)) for c, v in self.object_class_dict_p_rev.items()}

# 		self.object_dict = {i: self.object_dict_p[k] for i, k in enumerate(list(self.object_dict_p.keys()))} #object id to jpg paths #DO we need this?
# 		self.object_class_dict = {i: self.object_class_dict_p[k] for i, k in enumerate(list(self.object_dict_p.keys()))} #DO we need this?
# 		self.views = views
# 		self.transform = transform
# 		self.resize_transform = transforms.Resize((resize_shape, resize_shape))
# 		self.t = transforms.ToTensor()
# 		del self.object_dict_p; del self.object_class_dict_p

# 		#get the number of unique classes
# 		######################################################################
# 		#These are the added parts


# 		self.img_dict = {}
# 		img_counter = 0
# 		o_counter = 0
# 		self.class_idx2start = {c: None for c in self.class2idx}
# 		self.obj_idx2start = {}
# 		for c in self.class2idx:
# 			o_list = self.object_class_dict_p_rev[c]
# 			for o in o_list:
# 				imgs = self.object_dict[o]
# 				for img in imgs:
# 					self.img_dict[img_counter] = 
# 					img_counter +=1
# 				o_counter +=1

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

			
# 			if self.transform is not None:
# 				image = self.transform(image)

# 			if self.transform is None:
# 				image = self.resize_transform(image)
# 				image = self.t(image)

# 			return_dict['image_' + str(v)] = image
# 		return_dict['object_label'] = idx
# 		return_dict['category_label'] = self.class2idx[self.object_class_dict[idx]]

# 		return return_dict

