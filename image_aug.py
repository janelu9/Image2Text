#!/usr/bin/env python
# coding: utf-8
# Created on Mon Apr 11 16:55:39 2022
# @author: Lu Jian
# Email:janelu@live.cn; lujian@sdc.icbc.com.cn

from PIL import Image, ImageFilter
import numpy as np
from numpy.random import uniform,random

np.random.seed(2022)
from PIL import Image, ImageFilter
import numpy as np
from numpy.random import uniform,random

np.random.seed(2022)
class RandomPad:
    def __init__(self,LR=20,UL=20,w=0.5):
        self.LR=LR
        self.UL=UL
        self.w=w
    def __call__(self,img):
        img_arr = np.array(img)
        if random()>self.w:
            upper= img_arr[:1,:,:]
            img_arr = np.concatenate([np.tile(upper,(int(uniform()*self.UL),1,1)),img_arr],0)
        if random()>self.w:
            lower = img_arr[-1:,:,:]
            img_arr = np.concatenate([img_arr,np.tile(lower,(int(uniform()*self.UL),1,1))],0)
        if random()>self.w:
            left = img_arr[:,:1,:]
            img_arr = np.concatenate([np.tile(left,(1,int(uniform()*self.LR),1)),img_arr],1)
        if random()>self.w:
            right = img_arr[:,-1:,:]
            img_arr = np.concatenate([img_arr,np.tile(right,(1,int(uniform()*self.LR),1))],1)
        return  Image.fromarray(img_arr)

class GaussianBlur:
    def __init__(self, p=2.5):
        self.p = p
    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(uniform()*self.p))
    
class MinFilter:
    def __init__(self, p=3):
        self.p = p
    def __call__(self, image):
        return image.filter(ImageFilter.MinFilter(self.p))
    
class MaxFilter:
    def __init__(self, p=3):
        self.p = p
    def __call__(self, image):
        return image.filter(ImageFilter.MaxFilter(self.p))
    
class Rotate:
    def __init__(self, p=6):
        self.p=p
    def __call__(self,image):
        return image.rotate(uniform(-1,1)*self.p,fillcolor=(222,222,222))
    
class Resize:
    def __init__(self, w=384,h=384):
        self.w=w
        self.h=h
    def __call__(self,image):
        return image.resize((self.w,self.h))
    
class Normalize:
    def __init__(self, mean=0.5,std=0.5):
        if not isinstance(mean,(tuple,list)):
            mean=[mean]*3
            std=[std]*3
        self.mean=np.array([[[mean]]])
        self.std=np.array([[[std]]])
    def __call__(self,img_arr):
        return (img_arr/255 - self.mean)/self.std
    
class image_process:
    def __init__(self,size=384,aug = True):
        self.resize=Resize(size,size)
        self.normalize=Normalize()
        self.aug_flag = aug
        if self.aug_flag:
            self.aug=(
                (RandomPad(),0.8),
                (GaussianBlur(),0.5),
                (MinFilter(),0.3),
                (MaxFilter(),0.5),
                (Rotate(),0.8),
            )
    def infer_process(self,img):
        img=self.resize(img)
        img_arr = np.array(img)
        return img_arr

    def aug_process(self,img):
        for f,w in self.aug:
            if random()<w:
                img=f(img)
        return self.infer_process(img)
    
    def __call__(self,img):
        return self.aug_process(img) if self.aug_flag else self.infer_process(img)