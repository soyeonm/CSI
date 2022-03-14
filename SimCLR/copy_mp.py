import shutil
import multiprocessing 
import os
import cv2
from glob import glob
import subprocess



def copy_instance(file, dest, rank): 
    # printing process id to SHOW that we're actually using MULTIPROCESSING 
    if rank %1000 ==0:
    	print("ID of main process: {}".format(os.getpid()))     
    im = cv2.imread(file)
    im = cv2.resize(im, (300,300))
    #Make directory
    bashCommand = "mkdir -p " + os.path.join(dest,'/'.join(file.split('/')[:-1]))
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    #
    cv2.imwrite(os.path.join(dest,file), im)
    
    
    

if __name__ == "__main__": 
	assert os.getcwd() == '/projects/rsalakhugroup/soyeonm/co3d/co3d_download'
	#files = os.listdir(src) # Getting the files to copy
	#masks = glob('*/*/masks')
	objs = glob('*/*') #'baseballbat/375_42661_85494'
	#baseball_masks = glob(objs[0] + '/masks/*')
	#
	rank = 0
	for obj in objs:
		masks =  glob(obj + '/masks/*.png')
		for mask in masks:
			#p1 = multiprocessing.Process(target=copy_instance, args=(mask, '/projects/rsalakhugroup/soyeonm/co3d/copy_test', rank)) 
			p1 = multiprocessing.Process(target=copy_instance, args=(mask, '/home/soyeonm/projects/devendra/CSI/CSI_my/data/co3d_march_14_masks', rank)) 
			p1.start() 
			rank +=1
	    	