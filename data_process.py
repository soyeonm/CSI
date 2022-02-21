#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 12:26:04 2022

@author: soyeonmin
"""
import subprocess
from glob import glob
import numpy as np
zips = glob('*.zip')

for z in zips:
    bashCommand = "jar xf " + z 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    
#randomly select 20% and put into test, rest into training
from glob import glob



#Remove .json and .jgz

json_paths = glob('*/*.json')
jgz_paths = glob('*/*.jgz')

for p in json_paths + jgz_paths:
    bashCommand = "rm " + p 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

#Remove

object_folders = glob('*/*')
classes = set([o.split('/')[0] for o in object_folders])

for i, c in enumerate(list(classes)):
    c_object_folders = glob(c+ '/*')
    np.random.seed(i)
    perm = np.random.permutation(len(c_object_folders))
    twenty_num = int(0.2*len(c_object_folders))
    first_twenty = perm[:twenty_num].tolist()
    
    #bashCommand = "mkdir -p test/" + c 
    #process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    #output, error = process.communicate()
    
    for t in first_twenty:
        bashCommand = "mkdir -p test/" + c_object_folders[t]
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        
        bashCommand = "mv " + c_object_folders[t] + " " + "test/" + c_object_folders[t]
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
    
    
    

np.random.seed()

#generate random permutation and take the first 20%


#Expand into class:images

train_class_folders = glob('co3d_small_split_one/train/*')
test_class_folders = glob('co3d_small_split_one/test/*')
ood_class_folders = glob('co3d_small_split_one/ood/*')

for cf in train_class_folders+ood_class_folders:
    obj_folder = glob(cf + '/*')
    for of in obj_folder:
        im_folders = glob(of + '/*')
        for im_folder in im_folders:
            if im_folder.split('/')[-1] == 'images':
                new_im_folder = '/'.join(im_folder.split('/')[:-2]) #This will be something like 'co3d_small_split_one/ood/couch/'
                new_im_folder = new_im_folder.replace('co3d_small_split_one', 'co3d_small_split_one_no_by_obj') #this will be something like 'co3d_small_split_one_no_by_obj/ood/couch'
                #Make directory
                bashCommand = "mkdir -p " + new_im_folder 
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                
                #copy
                for im in glob(im_folder + '/*'):
                    new_im = 'obj_' + im.split('/')[-3] + '_' + im.split('/')[-1]  #must be something like 'obj_105_12605_26413_frame000049.jpg'
                    bashCommand = "cp " + im + " " + new_im_folder + "/" + new_im
                    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                    output, error = process.communicate()
                    
for cf in test_class_folders:
    obj_folder = glob(cf + '/*')
    for of in obj_folder:
        of = of + '/' + of.split('/')[-1]
        im_folders = glob(of + '/*')
        for im_folder in im_folders:
            #im_folder = im_folder + '/' + im_folder.split('/')[-1]
            if im_folder.split('/')[-1] == 'images':
                new_im_folder = '/'.join(im_folder.split('/')[:-2]) #This will be something like 'co3d_small_split_one/ood/couch/'
                new_im_folder = new_im_folder.replace('co3d_small_split_one', 'co3d_small_split_one_no_by_obj') #this will be something like 'co3d_small_split_one_no_by_obj/ood/couch'
                #Make directory
                bashCommand = "mkdir -p " + new_im_folder 
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                
                #copy
                for im in glob(im_folder + '/*'):
                    new_im = 'obj_' + im.split('/')[-3] + '_' + im.split('/')[-1]  #must be something like 'obj_105_12605_26413_frame000049.jpg'
                    bashCommand = "cp " + im + " " + '/'.join(new_im_folder.split('/')[:-1]) + "/" + new_im
                    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                    output, error = process.communicate()
                

#Process imagenet-30 (original) file names
import os
imagenet_test_folder = '/Users/soyeonmin/Downloads/one_class_test/*/*/*'
new_imagenet_test_folder = '/Users/soyeonmin/Downloads/one_class_test_dirnamechanged'


im_ori_test_class_folders = glob('/Users/soyeonmin/Downloads/one_class_test/*')

im_ori_test_imgs = glob(imagenet_test_folder)

for cl_f in im_ori_test_class_folders: 
    class_name = cl_f.split('/')[-1]
    clas_write_folder = os.path.join(new_imagenet_test_folder, class_name)
    bashCommand = "mkdir -p " + clas_write_folder
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    for subf in glob(os.path.join(cl_f, '*')): #subf looks like '/Users/soyeonmin/Downloads/one_class_test/parking_meter/n03891332'
        for im_f in glob(os.path.join(subf, '*')): #imf looks like '/Users/soyeonmin/Downloads/one_class_test/parking_meter/n03891332/47.JPEG'
            last_im_f  = im_f.split('/')[-1]
            subfolder = im_f.split('/')[-2]
            new_im_f = subfolder + "_" + last_im_f
            new_im_path = os.path.join(clas_write_folder, new_im_f)
            
            bashCommand = "cp " + im_f + " " + new_im_path
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
    
    
#Now resize original imagenet-30 and save
import os

import cv2
imagenet_test_folder = '/Users/soyeonmin/Downloads/one_class_test/*/*/*'

resize_imagenet_test_folder = '/Users/soyeonmin/Downloads/one_class_test_oriimgnet_resized_to_32_by_me'

im_ori_test_class_folders = glob('/Users/soyeonmin/Downloads/one_class_test/*')

im_ori_test_imgs = glob(imagenet_test_folder)

for cl_f in im_ori_test_class_folders: 
    class_name = cl_f.split('/')[-1]
    clas_write_folder = os.path.join(resize_imagenet_test_folder, class_name)
    bashCommand = "mkdir -p " + clas_write_folder
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    for subf in glob(os.path.join(cl_f, '*')): #subf looks like '/Users/soyeonmin/Downloads/one_class_test/parking_meter/n03891332'
        for im_f in glob(os.path.join(subf, '*')): #imf looks like '/Users/soyeonmin/Downloads/one_class_test/parking_meter/n03891332/47.JPEG'
            last_im_f  = im_f.split('/')[-1]
            subfolder = im_f.split('/')[-2]
            new_im_f = subfolder + "_" + last_im_f
            new_im_path = os.path.join(clas_write_folder, new_im_f)
            
            im = cv2.imread(im_f)
            im = cv2.resize(im, (32,32))
            
            cv2.imwrite(new_im_path, im)
            
            
#
#Make new ood with less confusing data with hydrant, toilet, apple
#
folders = ['apple', 'hydrant', 'toilet']
for cf in folders:
    obj_folder = glob(cf + '/*')
    for of in obj_folder:
        im_folders = glob(of + '/*')
        for im_folder in im_folders:
            if im_folder.split('/')[-1] == 'images':
                new_im_folder = '/'.join(im_folder.split('/')[:-2]) #This will be something like 'co3d_small_split_one/ood/couch/'
                #new_im_folder = new_im_folder.replace('co3d_download_folder_fifth', 'new_ood_apple_hydrant_toilet') #this will be something like 'co3d_small_split_one_no_by_obj/ood/couch'
                new_im_folder = '/home/soyeonm/novelty_detection_datasets/co3d/new_ood_apple_hydrant_toilet/'+ new_im_folder
                #Make directory
                bashCommand = "mkdir -p " + new_im_folder 
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                
                #copy
                for im in glob(im_folder + '/*'):
                    new_im = 'obj_' + im.split('/')[-3] + '_' + im.split('/')[-1]  #must be something like 'obj_105_12605_26413_frame000049.jpg'
                    bashCommand = "cp " + im + " " + new_im_folder + "/" + new_im
                    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                    output, error = process.communicate()
                    
            
#
#Make new train/ test split with large amount of data
#
import subprocess
from glob import glob
import numpy as np
classes = ['hairdryer', 'suitcase', 'broccoli']

#First split into train and test
for i, c in enumerate(list(classes)):
    c_object_folders = glob(c+ '/*')
    np.random.seed(i)
    perm = np.random.permutation(len(c_object_folders))
    twenty_num = int(0.2*len(c_object_folders))
    first_twenty = perm[:twenty_num].tolist()
    
    #bashCommand = "mkdir -p test/" + c 
    #process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    #output, error = process.communicate()
    
    for t in first_twenty:
        bashCommand = "mkdir -p test/" + c_object_folders[t]
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        
        bashCommand = "mv " + c_object_folders[t] + " " + "test/" + c_object_folders[t]
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        
#Rewrite to train, test
#Test
for i, c in enumerate(list(classes)):
    c_object_folders = glob('test/' + c+ '/*/*/images')
    bashCommand = "mkdir -p largerco3d/test/" + c
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    for cof in c_object_folders:
        imgs = glob(cof+ "/*.jpg")
        for im in imgs:
            new_name = 'obj_' + cof.split('/')[-2] + '_' + im.split('/')[-1]
            
            bashCommand = "mv " + im + " largerco3d/test/" + c + "/" +new_name
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

#Train 
for i, c in enumerate(list(classes)):
    c_object_folders = glob(c+ '/*/images')
    bashCommand = "mkdir -p largerco3d/train/" + c
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    for cof in c_object_folders:
        imgs = glob(cof+ "/*.jpg")
        for im in imgs:
            new_name = 'obj_' + cof.split('/')[-2] + '_' + im.split('/')[-1]
            
            bashCommand = "mv " + im + " largerco3d/train/" + c + "/" + new_name
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate() 