#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 10:44:11 2018

@author: yinxueqi
"""

import random    
import os   
import os.path as osp    
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
from labelme import utils

dataset_dir='/home/jiangmignzhi/FCN.tensorflow-master/label/annotation_img2'
  
def file_name(file_dir):       
    L=[]   
   
    for root, dirs, files in os.walk(file_dir):      
        for file in files:      
            if os.path.splitext(file)[1] == '.json':      
                #L.append(os.path.join(root, file))    
                L.append(file) 
        
    return L    

my_filename=file_name(dataset_dir)
my_filename.sort()

for i in my_filename:
   
    #print(i)
    data = json.load(open(os.path.join(dataset_dir, i)))
    img = utils.img_b64_to_array(data['imageData'])
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
    
    captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
    lbl_viz = utils.draw_label(lbl, img, captions)
    
    # lbl_names[0] 默认为背景，对应的像素值为0
    # 解析图片中的对象 像素值不为0（0 对应背景）
    mask=[]
    class_id=[]
    for j in range(1,len(lbl_names)): # 跳过第一个class（默认为背景）
        mask.append((lbl==j).astype(np.uint8)) # 解析出像素值为1的对应，对应第一个对象 mask 为0、1组成的（0为背景，1为对象）
        class_id.append(j) # mask与clas 一一对应
    
    mask=np.transpose(np.asarray(mask,np.uint8),[1,2,0]) # 转成[h,w,instance count]
    class_id=np.asarray(class_id,np.uint8) # [instance count,]
     #   class_name=lbl_names[1:] # 不需要包含背景
    
    # plt.imshow(mask[:,:,0],'gray')
#   savename = os.path.join(i,'1') 
    plt.axis('off')
    #savename = os.path.join(i,'1') 
    #plt.savefig('lena_new_sz.png')
    str1 = str(i)
    rm = '.json'
    str2 = str1.rstrip(rm)   
    #print(str2)
    # plt.savefig(str2+'.png')
    im = mask[:,:,0]
    img = Image.fromarray(im, 'L')
    img.save(os.path.join('/home/jiangmignzhi/FCN.tensorflow-master/label/after', str2+'.png'), 'png')
    
    
    
    
    
    
    
    
    
    
    
    