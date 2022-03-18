#copy_mp with just 1000 workers

import shutil
import multiprocessing 
import os
import cv2
from glob import glob
import subprocess
import pickle

def copy_instance(files, dest, rank): 
	# printing process id to SHOW that we're actually using MULTIPROCESSING 
	if rank %100 ==0:
		print("ID of main process: {}".format(os.getpid()))   
	for file in files:  
		#print("dest is ", dest)
		#print("file is ", file)
		im = cv2.imread('/projects/tir6/bisk/soyeonm/projects/co3d/download/' + file)
		#im = cv2.imread(file)
		im = cv2.resize(im, (300,300))
		#Make directory
		bashCommand = "mkdir -p " + os.path.join(dest,'/'.join(file.split('/')[:-1]))
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()
		#
		cv2.imwrite(os.path.join(dest,file), im)#; print("joined is ", os.path.join(dest,file))

if __name__ == "__main__": 
	num_workers = 100
	#os.chdir('/projects/rsalakhugroup/soyeonm/co3d/co3d_download')
	imgs_train = pickle.load(open('train_globs.p', 'rb'))
	imgs_test = pickle.load(open('test_globs.p', 'rb'))

	imgs = [img.replace('/projects/tir6/bisk/soyeonm/projects/co3d/download/', '') for img in imgs_train + imgs_test] #Do this replace or do change os
	masks = [img.replace('images', 'masks').replace('.jpg', '.png') for img in imgs]
	#os.chdir('/projects/tir6/bisk/soyeonm/projects/co3d/download')
	#assert os.getcwd() == '/projects/rsalakhugroup/soyeonm/co3d/co3d_download'
	#assert os.getcwd() == '/projects/tir6/bisk/soyeonm/projects/co3d/download'
	#files = os.listdir(src) # Getting the files to copy
	#masks = glob('*/*/masks/*.png')
	#imgs = [m.replace('masks', 'images').replace('.png', '.jpg') for m in masks]

	#Divide it evenly into num_workers
	#objs = glob('*/*') #'baseballbat/375_42661_85494'
	#baseball_masks = glob(objs[0] + '/masks/*')
	#
	for worker_i in range(num_workers):
		if worker_i < num_workers - 1:
			masks_i = masks[worker_i* int(len(masks)/num_workers):(worker_i+1)*int(len(masks)/num_workers)]
			imgs_i = imgs[worker_i* int(len(masks)/num_workers):(worker_i+1)*int(len(masks)/num_workers)]
		else:
			masks_i = masks[worker_i* int(len(masks)/num_workers):]
			imgs_i = imgs_i[worker_i* int(len(masks)/num_workers):]
		#p1 = multiprocessing.Process(target=copy_instance, args=(masks_i + imgs_i, '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_14_masks', worker_i)) 
		p1 = multiprocessing.Process(target=copy_instance, args=(masks_i + imgs_i, '/home/soyeonm/CSI/CSI_my/data/co3d_masks_imgs_March_17', worker_i)) 
		p1.start() 
