#!/usr/bin/env python
# coding: utf-8
# Created on Mon Apr 11 16:55:39 2022
# @author: Lu Jian
# Email:janelu@live.cn;

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from numpy.random import uniform,random

np.random.seed(2022)

class RandomPad:
    def __init__(self,LR=15,UL=15,w=0.5):
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

class RandomCut:
    def __init__(self, LR=14, UL=14, w=0.5):
        self.LR=LR
        self.UL=UL 
        self.w=w
    def __call__(self, img_arr) :
        img_arr = np.array(img_arr)
        if random()>self.w:
            img_arr =img_arr[:min(-1,-int(random()*self.UL)),:,:]
        if random()>self.w:
            img_arr =img_arr[int(random()*self.UL):,:,:] 
        if random()>self.w:
            img_arr =img_arr[:,:min(-int(random()*self.LR),-1),:]
        if random()>self.w:
            img_arr =img_arr[:,int(random()*self.LR):,:] 
        return img_arr

class RandomMask:
    def __init__(self,fill=0,space=(10,30),p=0.8):
        self.fill=fill
        self.space=space
        self.p=p
    def __call__(self,img_arr):
        if random()<self.p:
            img_arr= np.array(img_arr)
            H=img_arr.shape[0]
            h = randint(*self.space)
            a = randint(0,high=H-h)
            img_arr[a:a+h,a:a+h,:]=self.fill
        return img_arr


class Bright:
    def __init__(self, a=0.5,b=1.5):
        self.a = a
        self.b = b
    def __call__(self, image):
        Enhancer = ImageEnhance.Brightness(image)
        return Enhancer.enhance(uniform(self.a,self.b))
        
class Contrast:
    def __init__(self,a=0.3, b=1.3):
        self.a = a
        self.b = b
    def __call__(self, image):
        Enhancer = ImageEnhance.Contrast(image)
        return Enhancer.enhance(uniform(self.a,self.b))
        
class Color:
    def __init__(self,a=0.3, b=3):
        self.a = a
        self.b = b
    def __call__(self, image):
        Enhancer = ImageEnhance.Color(image)
        return Enhancer.enhance(uniform(self.a,self.b))
        
class Sharpness:
    def __init__(self,a=0, b=5):
        self.a = a
        self.b = b
    def __call__(self, image):
        Enhancer = ImageEnhance.Sharpness(image)
        return Enhancer.enhance(uniform(self.a,self.b))
        
class GaussianBlur:
    def __init__(self, p=1.5):
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
        self.mean=np.array([[[mean]]],'float32')
        self.std=np.array([[[std]]],'float32')
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
                (RandomCut(),0.8),
                (Bright(),0.3),
                (Contrast(),0.3),
                (Sharpness(),0.3),
                (Color(),0.3),
                (GaussianBlur(),0.5),
                (MinFilter(),0.2),
                (MaxFilter(),0.3),
                (Rotate(),0.5),
            )
            self.rm=RandomMask()
    def infer_process(self,img):
        img=self.resize(img)
        if self.aug_flag:
            img=self.rm(img)
        img_arr = np.array(img,'float32')
        return img_arr

    def aug_process(self,img):
        for f,w in self.aug:
            if random()<w:
                img=f(img)
        return img
    
    def __call__(self,img):
        if self.aug_flag :
            img = self.aug_process(img)
        return  self.infer_process(img)