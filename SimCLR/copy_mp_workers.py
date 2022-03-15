#copy_mp with just 1000 workers

import shutil
import multiprocessing 
import os
import cv2
from glob import glob
import subprocess

def copy_instance(files, dest, rank): 
    # printing process id to SHOW that we're actually using MULTIPROCESSING 
    if rank %1000 ==0:
    	print("ID of main process: {}".format(os.getpid()))   
   	for file in files:  
	    im = cv2.imread(file)
	    im = cv2.resize(im, (300,300))
	    #Make directory
	    bashCommand = "mkdir -p " + os.path.join(dest,'/'.join(file.split('/')[:-1]))
	    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	    output, error = process.communicate()
	    #
	    cv2.imwrite(os.path.join(dest,file), im)

if __name__ == "__main__": 
	num_workers = 10000
	assert os.getcwd() == '/projects/rsalakhugroup/soyeonm/co3d/co3d_download'
	#files = os.listdir(src) # Getting the files to copy
	masks = glob('*/*/masks/*.png')
	#Divide it evenly into num_workers
	each_worker
	#objs = glob('*/*') #'baseballbat/375_42661_85494'
	#baseball_masks = glob(objs[0] + '/masks/*')
	#
	for worker_i in range(num_workers):
		if worker_i < num_workers = 1:
			masks_i = masks[worker_i* int(len(masks)/num_workers):(worker_i+1)*int(len(masks)/num_workers)]
		else:
			masks_i = masks[worker_i* int(len(masks)/num_workers):]
		p1 = multiprocessing.Process(target=copy_instance, args=(masks, '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_14_masks', rank)) 
