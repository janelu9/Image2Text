# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 21:28:02 2022
@author: Lu Jian
Email:janelu@live.cn; lujian@sdc.icbc.com.cn
"""

from PIL import Image, ImageFilter
import numpy as np
from numpy.random import uniform,random

class RandomPad:
    def __init__(self,LR=6,UL=15,w=0.5):
        self.LR=LR
        self.UL=UL
        self.w=w
    def __call__(self,img_arr):
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
        return img_arr

class GaussianBlur:
    def __init__(self, p=2.5):
        self.p = p
    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(p=self.p))
    
class MinFilter:
    def __init__(self, p=3):
        self.p = p
    def __call__(self, image):
        return image.filter(ImageFilter.MinFilter(p=self.p))
    
class MaxFilter:
    def __init__(self, p=3):
        self.p = p
    def __call__(self, image):
        return image.filter(ImageFilter.MaxFilter(p=self.p))
    
class Rotate:
    def __init__(self, p=6):
        self.p=p
    def __call__(self,image):
        return image.rotate(uniform(-1,1)*self.p,fillcolor=(222,222,222))
    
class Resize:
    def __init__(self, w=224,h=224):
        self.w=w
        self.h=h
    def __call__(self,image):
        return image.resize((self.w,self.h))
