#!/usr/bin/env python
# coding: utf-8
import sys
import tensorflow as tf
from collections import defaultdict
import os

from DeepSUM_network import SR_network
from PIL import Image
import math

import json

config_file='/home/db129/SuperResolution/DeepSUM_varloss/config_files/DeepSUM_normalised.json'
##config_file=r'C:\DeepLearning\COMPX594\DeepSUM\DeepSUM\config_files\DeepSUM_config_NIR_SentinelWRAPS_2.json'

with open(config_file) as json_data_file:
    data = json.load(json_data_file)

band =  'red'
print (band)
        
tf.reset_default_graph()
config=defaultdict()
config['lr']= data['hyperparameters']['lr']
config['batch_size'] =  data['hyperparameters']['batch_size']
config['base_dir'] = data['others']['base_dir']
config['skip_step'] = data['others']['skip_step']
config['channels'] = data['others']['channels']
config['T_in'] = data['others']['T_in'] 
config['R'] = data['others']['R']
config['Im'] = data['others']['Im']
config['full'] = data['others']['full']
config['patch_size_HR'] = data['others']['patch_size_HR']
config['patch_size_LR'] = data['others']['patch_size_LR']
config['border'] = data['others']['border']
config['spectral_band']=data['others']['spectral_band']
config['RegNet_pretrain_dir']=data['others']['RegNet_pretrain_dir']
config['SISRNet_pretrain_dir']=data['others']['SISRNet_pretrain_dir']
config['dataset_path']=data['others']['dataset_path']
config['n_chunks']=data['others']['n_chunks']
config['mu']=data['others']['mu']
config['sigma']=data['others']['sigma']
config['sigma_rescaled']=data['others']['sigma_rescaled']

config['tensorboard_dir'] = 'green'
#print (config['mu'])
model = SR_network(config)

model.build()

def makenostr(i):
    numi = str(i)
    if len(numi) == 1:
        numi = '000' + numi
    if len(numi) == 2:
        numi = '00' + numi
    if len(numi) == 3:
        numi = '0' + numi
    return numi

##dir_test=r'C:\DeepLearning\COMPX594\Data\testset'

dir_test =r'/Scratch/db129/rgb_all_test/'
location = '/home/db129/SuperResolution/TestImagesDir/TEST_varloss3' ##output location

r=0
g=0
b=0
##iterate through chunks and super-resolve
for dird in  sorted(os.listdir(dir_test)):
  #r=int(math.ceil(r / 200.0)) * 200
  #g=int(math.ceil(g / 200.0)) * 200
  #b=int(math.ceil(b / 200.0)) * 200
  if dird[:3] == 'red':
    band = 'red'
    mu=85.9
    sigma=28.5
    numi = makenostr(r)
    
  elif dird[:3] == 'blu':
    band = 'blue'
    mu=99.4
    sigma=25.7
    numi = makenostr(b)
    
  elif dird[:3] == 'gre':
    band = 'green'
    mu=98.4
    sigma=25.7
    numi = makenostr(g)
    
    
  config['tensorboard_dir'] = band
  i=0
  n_slide=0
  super_resolved_images, LRmap=model.predict_test(dir_test + '/' + dird,n_slide=n_slide,setsize=6,mu=mu,sigma=sigma,band=band)
  for img in super_resolved_images:
      while i in LRmap:
        print(str(i) + " is in LRMap")
        i=i+1
        if band=='red':
          r=r+1
          numi = makenostr(r)
        elif band=='green':
          g=g+1
          numi = makenostr(g)
        elif band=='blue':
          b=b+1
          numi = makenostr(b)
      
      print(str(i) + " is NOT in LRMap")
      img1 = img[0,:,:,0]
      new_img = Image.fromarray(img1.astype('uint8'),'L')
      ##new_img = Image.fromarray(normalize(img1).astype('uint8'),'L')
      #image = Image.fromarray(img1, 'L')
      
      new_img.save(location + '/' + numi + '_' + band + '.png')
      print('Save Image number {0} , {1}'.format(band, numi))
      i=i+1
      if band=='red':
        r=r+1
        numi = makenostr(r)
      elif band=='green':
        g=g+1
        numi = makenostr(g)
      elif band=='blue':
        b=b+1
        numi = makenostr(b)
        
for i in range(0,100):
    numi = makenostr(i)
    print (numi)
    if os.path.exists(location + '/' + numi +'_red.png') and os.path.exists(location + '/' + numi +'_blue.png') and os.path.exists(location + '/' + numi +'_green.png'):
      red = Image.open(location + '/' + numi +'_red.png').convert('L')
      blue = Image.open(location + '/' + numi +'_blue.png').convert('L')
      green = Image.open(location + '/' + numi +'_green.png').convert('L')
      out = Image.merge("RGB", (red, green, blue))
      out.save(location + '/' + 'rgb' + numi + '.png')
      

  
