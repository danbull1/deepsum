##

import os
import glob
import numpy as np
from skimage import io 
from PIL import Image,ImageStat
import math 
import cv2

import sys
sys.path.insert(0, '/home/db129/SuperResolution/DeepSUM/libraries/')
#sys.path.append(r'C:\DeepLearning\COMPX594\DeepSUM\DeepSUM\libraries')

from dataloader import load_dataset,create_patch_dataset_return_shifts
from utils import safe_mkdir,upsampling_without_aggregation_all_imageset,upsampling_mask_all_imageset,registration_imageset_against_best_image_without_union_mask
import numpy as np
import pandas as pd
from skimage.feature import register_translation

chunksize=200


base = '/Scratch/db129/IM'

out = '/Scratch/db129/rgb_all2'

def getstats(dir_list,  band):
    firstimageset = dir_list[0]

    imagestats = {}
    images = []
    for filename in os.listdir(firstimageset):
        if filename[:3] == band[:3]:
            images.append(filename.replace(band,"")[5:-4])

    for image in images[:50]:
        means = []
        var = []
        for folder in os.listdir(base):
            for filename in os.listdir(base + '/' + folder):
                #if folder == 'imgset109':
                #    print ("109")
                if filename.replace(band,"")[5:-4] == image:
                    file = base + '/' + folder + '/'  + filename
                    cloudname = filename.replace(band,"cloud")
                    cloud = base + '/' + folder + '/' + cloudname
                    img = Image.open(file)
                    cld = Image.open(cloud)
                    img_arr = np.array(img)
                    cld_arr = np.array(cld)
                    if cld_arr.sum() >1 and img_arr.sum() >1:
                        masked = img_arr*cld_arr
                        if masked.sum() >1:
                            masked_img = np.ma.masked_where(masked==0, masked)#img_arr*cld_arr
                            mean = masked_img.mean()
                            if not mean >1:
                                print (str(mean) + " " + folder)
                            variance = masked_img.var()
                            means.append(mean)
                            var.append(variance)
        meanall = sum(means)/len(means)
        stdev = math.sqrt(sum(var)/len(var))
        print (meanall)
        print (stdev)
        imagestats[image] = [meanall, stdev]
    return imagestats

def applystats(image, imagename, imagestats, band):
    imagename = imagename.split('/')[-1].replace(band,"")[5:-4]
    #imagename = imagename.replace(imagename[4:8], "0000")

    mean = imagestats[imagename][0]
    stdev = imagestats[imagename][1]

    normimage =  (image-mean)/stdev
    return normimage

   

def load_from_directory_to_pickle(base_dir,out_dir,band, chunksizefrom, chunksizeto, imagestats):

    dir_list=sorted(glob.glob(base_dir+'/imgset*'))
    dir_list = dir_list[chunksizefrom:chunksizeto]
    dir_list.sort()

    #imagestats = getstats(dir_list, band)

    test_images_LR = np.array([[applystats(io.imread(fname),fname, imagestats, band) for fname in sorted(glob.glob(dir_name+'/' + band + '*.png'))] 
                             for dir_name in dir_list ]) ##DJB updated from io.imread(fname,dtype=np.uint16)

    #test_images_LR.dump(out_dir+'/'+'LR_test_'+band+'.npy')
    

    test_mask_LR = np.array([[io.imread(fname).astype(np.bool) for fname in sorted(glob.glob(dir_name+'/cloud*.png'))] 
                             for dir_name in dir_list ])

    return test_images_LR, test_mask_LR
    
    #test_mask_LR.dump(out_dir+'/'+'LR_mask_'+band+'_test.npy')

def processpickles(images, masks, band, out_dataset):
    setsize=6
    #input_images_LR_test = np.load(os.path.join(dir_pickles, 'LR_test_{0}.npy'.format(band)),allow_pickle=True)
    #mask_LR_test = np.load(os.path.join(dir_pickles, 'LR_mask_{0}_test.npy'.format(band)),allow_pickle=True)

    #transform in a list of numpy
    input_images_LR_test=np.array([np.array(x) for x in images])
    mask_LR_test=np.array([np.array(x) for x in masks])

    ##NOT Removing all images in which we find at least one pixel > 60000.

    input_images_LR_test_upsample=upsampling_without_aggregation_all_imageset(input_images_LR_test,scale=4)
    mask_LR_test_upsample=upsampling_mask_all_imageset(mask_LR_test,scale=4)
    input_images_LR_test_upsample_registered,mask_LR_test_upsample_registered,shifts_test,new_index_orders_test=registration_imageset_against_best_image_without_union_mask(input_images_LR_test_upsample,
                                                    mask_LR_test_upsample,1)

    #transform in a list of numpy
    input_images_LR_test_upsample=np.array([np.array(x) for x in input_images_LR_test_upsample])
    mask_LR_test_upsample=np.array([np.array(x) for x in mask_LR_test_upsample])

    #Reorder the upsampled and not registered testset the way the upsdampled and registered testset has been ordered
    #so that it matched the ordering of the shifts we computed during registration
    input_images_LR_test_upsample=[imageset[new_index_orders_test[i]] for i,imageset in enumerate(input_images_LR_test_upsample)]
    mask_LR_test_upsample=[imageset[new_index_orders_test[i]] for i,imageset in enumerate(mask_LR_test_upsample)]

    #Find the indexes to remove considering we want to keep up to 4 pixel shift.
    images_to_remove=[[i,j,z] for i,x in enumerate(shifts_test) for j,z in enumerate(x) if (np.abs(z)>4).any() ]
    #generate dictionary with as key the index of the imageset and as value a list of indexes correpsonding to images
    # to remove of that specific imageset
    from collections import defaultdict

    d=defaultdict(list)

    for i in images_to_remove:
        d[i[0]].append(i[1])

    for i in d.keys():
        input_images_LR_test_upsample[i]=np.delete(input_images_LR_test_upsample[i],d[i],axis=0)
        mask_LR_test_upsample[i]=np.delete(mask_LR_test_upsample[i],d[i],axis=0)


    indexes=[i for i,x in enumerate(input_images_LR_test_upsample) if np.array(x).shape[0]<setsize]
    input_images_LR_test_upsample=np.delete(input_images_LR_test_upsample,indexes,axis=0)
    mask_LR_test_upsample=np.delete(mask_LR_test_upsample,indexes,axis=0)

    # Update also shifts array
    for i in d.keys():
        shifts_test[i]=np.delete(shifts_test[i],d[i],axis=0)

    shifts_test=np.delete(shifts_test,indexes,axis=0)

    ###########################DJB
    ##remove more than 3 images in set
    for i,x in enumerate(input_images_LR_test_upsample):
    
        toomanyimages = [j for j,z in enumerate(x) if j > setsize-1]
        x = np.delete(input_images_LR_test_upsample[i],toomanyimages,axis=0)
        input_images_LR_test_upsample[i]=np.delete(input_images_LR_test_upsample[i],toomanyimages,axis=0)
        mask_LR_test_upsample[i]=np.delete(mask_LR_test_upsample[i],toomanyimages,axis=0)
        shifts_test[i]=np.delete(shifts_test[i],toomanyimages,axis=0)

    np.save(os.path.join(out_dataset,'dataset_{0}_LRmap.npy'.format(band)),indexes,allow_pickle=True)
    np.save(os.path.join(out_dataset,'dataset_{0}_LR_test.npy'.format(band)),input_images_LR_test_upsample,allow_pickle=True)
    np.save(os.path.join(out_dataset,'dataset_{0}_mask_LR_test.npy'.format(band)),mask_LR_test_upsample,allow_pickle=True)
    np.save(os.path.join(out_dataset,'shifts_test_{0}.npy'.format(band)),shifts_test,allow_pickle=True)

chunksize=200
cnt = len(os.listdir(base))

iters = math.ceil(cnt/chunksize)
dir_list=glob.glob(base+'/imgset*')
imagestatsred = getstats(dir_list, 'red')
imagestatsblue = getstats(dir_list, 'blue')
imagestatsgreen = getstats(dir_list, 'green')
for i in range(0,5):
    num = str(i)
    if len(num)==1:
      num = '0' + num
    newgreendir = out + '/green_' + num
    os.mkdir(newgreendir)
    test_images_LR, test_mask_LR = load_from_directory_to_pickle(base, newgreendir, 'green', i*chunksize, (i+1)*chunksize, imagestatsgreen)
    processpickles(test_images_LR, test_mask_LR, 'GREEN', newgreendir)
    newreddir = out + '/red_' + num
    os.mkdir(newreddir)
    test_images_LR, test_mask_LR = load_from_directory_to_pickle(base, newreddir, 'red', i*chunksize, (i+1)*chunksize, imagestatsred)
    processpickles(test_images_LR, test_mask_LR, 'RED', newreddir)
    newbluedir = out + '/blue_' + num
    os.mkdir(newbluedir)
    test_images_LR, test_mask_LR = load_from_directory_to_pickle(base, newbluedir, 'blue', i*chunksize, (i+1)*chunksize, imagestatsgreen)
    processpickles(test_images_LR, test_mask_LR, 'BLUE', newbluedir)
    print ("saved " +str(i))
