#!/usr/bin/env python
# coding: utf-8
# Created on Mon Apr 11 20:18:46 2022
# @author: Lu Jian
# Email:janelu@live.cn; lujian@sdc.icbc.com.cn

from paddle.io import Dataset
from PIL import Image
import os
import numpy as np

class SimpleDataSet(Dataset):
    def __init__(self,img_pths,label_lists,image_process,tokenizer):
        super(SimpleDataSet, self).__init__()
        self.img_pths=img_pths.split(",") 
        self.label_lists=label_lists.split(",") 
        self.tokenizer=tokenizer
        self.image_process=image_process
        self.bos_id=self.pad_id=tokenizer.bos_token_id
        self.eos_id=tokenizer.eos_token_id
        self.data = self.gen_data()
        
    def gen_data(self):
        data=[]
        for d,l in zip(self.img_pths,self.label_lists):
            with open(l, 'r') as fp:
                for line in fp.readlines():
                    img_name,text=line.strip().split("\t")
                    img_path = os.path.join(d, img_name)
                    ids=self.tokenizer(text)["input_ids"]
                    data.append((img_path,ids))
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img,txt = self.data[idx]
        try:
            image = Image.open(img).convert('RGB')
            tfm_img = self.image_process(image)  # h, w, c
        except:
            rnd_idx = (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return { 'img': tfm_img, 'ids': txt}
    
    def collate_fn(self,x):
        d={'img':[],'tgt':[],'label':[]}
        max_len = max(len(i["ids"]) for i in x)
        for item in x:
            d['img'].append(item['img'])
            temp_id=item['ids']+[self.eos_id]+[self.pad_id]*(max_len-len(item['ids']))
            d['tgt'].append([self.bos_id]+temp_id)
            d['label'].append(temp_id+[self.pad_id])
        d['img']=self.image_process.normalize(np.array(d['img'])).transpose(0,3,1,2)
        d['tgt']=np.array(d['tgt'])
        d['label']=np.array(d['label'])
        return d