#!/usr/bin/env python
# coding: utf-8
# Created on Mon Apr 11 16:55:39 2022
# @author: Lu Jian
# Email:janelu@live.cn; lujian@sdc.icbc.com.cn

from data_loader import SimpleDataSet
from image_aug import image_process
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddlenlp.transformers import GPTChineseTokenizer
import paddle.distributed as dist
import os
from image2text import SwinTransformerEncoder,TransformerDecoder,Image2Text,WordEmbedding,PositionalEmbedding,FasterTransformer
from lr_scheduler import InverseSqrt
from argparse import ArgumentParser
import numpy as np

def parse_args():

    parser = ArgumentParser(description="Paddle distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    parser.add_argument("--decoder-pretrained", type=str,default="../gpt/gpt-cpm-small-cn-distill.pdparams",
                        help="pretrained model's params path")
    
    parser.add_argument("--data-dir", type=str,default="/mnt/e/OCR/unilm/trocr/IAM/image/",
                        help="data _dir")
    
    parser.add_argument("--train-list", type=str,default="/mnt/e/OCR/unilm/trocr/IAM/gt_test.txt",
                        help="train data dir")
                             
    parser.add_argument("--test-list", type=str,default="/mnt/e/OCR/unilm/trocr/IAM/gt_test.txt",
                        help="test data dir")                         
    return parser.parse_args()


def train(args):
    dist.init_parallel_env()
    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id)
    device = paddle.set_device(device)    
    tokenizer = GPTChineseTokenizer("gpt-cpm-cn-sentencepiece.model")    
    train_dataset=SimpleDataSet(args.data_dir,args.train_list,image_process(224),tokenizer)
    test_dataset=SimpleDataSet(args.data_dir,args.test_list,image_process(224,False),tokenizer)
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True, 
        batch_size=8,
        drop_last=False,
        places=device,
        num_workers=2,
        return_list=True,
        collate_fn = train_dataset.collate_fn,
        use_shared_memory=True)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False, 
        batch_size=2,
        drop_last=False,
        places=device,
        num_workers=2,
        return_list=True,
        collate_fn = test_dataset.collate_fn,
        use_shared_memory=True)
        
    encoder = SwinTransformerEncoder(img_size=224,embed_dim=48,depths=[2, 2, 6, 2],num_heads=[3, 6, 12, 24],window_size=7,drop_path_rate=0.2)
    decoder = TransformerDecoder(d_model=384,n_head=6,dim_feedforward=1536,num_layers=6)
    word_emb = WordEmbedding(vocab_size=tokenizer.vocab_size,emb_dim=decoder.d_model,pad_id=train_dataset.pad_id)
    pos_emb = PositionalEmbedding(decoder.d_model,max_length=512,learned=True)
    project_out = nn.Linear(decoder.d_model, word_emb.vocab_size)
    
    def load_pretrained_params(path):
        pretrained_model=paddle.load(path)
        word_emb.load_dict({'word_embeddings.weight':pretrained_model['embeddings.word_embeddings.weight']})
        pos_emb.load_dict({'position_embeddings.weight':pretrained_model['embeddings.position_embeddings.weight']})
        state_dict={}
        un_matched = []
        for k in decoder.state_dict().keys():
            if "decoder."+k in pretrained_model:
                state_dict[k] = pretrained_model["decoder."+k]
            else:
                un_matched.append(k)
        if len(un_matched)>0:
            print("unmatched keys:%s" % str(un_matched))
        decoder.load_dict(state_dict)
    
    load_pretrained_params(args.decoder_pretrained)
    model=Image2Text(encoder,decoder,word_emb,pos_emb,project_out,train_dataset.eos_id)
    fast_infer = FasterTransformer(model,max_out_len=70)
    
    model = paddle.DataParallel(model)
    model.train()
    scheduler = paddle.optimizer.lr.NoamDecay(d_model=decoder.d_model, warmup_steps=500, verbose=False)
    adam = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=scheduler,weight_decay= 0.0001)
    
    def metric(pred,label,tokenizer):
        if len(pred.shape)==3:
            pred=pred.argmax(-1)
        pred=pred.numpy()
        lable=label.numpy()
        m,n = pred.shape
        right = 0
        ignore={tokenizer.eos_token_id,tokenizer.bos_token_id}
        _,e=np.where(label==tokenizer.eos_token_id)
        for i in range(m):
            p=[]
            for j in range(n):
                t=int(pred[i][j])
                if t not in ignore: 
                    p.append(t)
                elif t== tokenizer.eos_token_id:
                    break
            p=tokenizer.convert_ids_to_string(p)
            l=tokenizer.convert_ids_to_string([int(j) for j in label[i][:e[i]]])
            if p==l:
                right+=1
        return right,m
        
    epochs=500
    batch_id=1
    R=S=L=0
    log_period=200
    test_period=1000
    train_acc=0
    test_acc=0.65
    for epoch in range(epochs):    
        for data in train_loader():
            predicts = model(data['img'],data['tgt'],tgt_mask=True)
            loss = paddle.nn.functional.cross_entropy(predicts, data['label'])
            right,samples = metric(predicts,data['label'],tokenizer)
            loss.backward()
            adam.step()
            adam.clear_grad()
            scheduler.step()
            
            batch_id+=1
            R += right
            S += samples
            L += loss.numpy()[0]
            if (batch_id) % log_period == 0:
                cur_train_period_acc = R/S
                print("epoch: {}, batch_id: {}, loss is: {}, lr is: {}, acc is: {}".\
                format(epoch, batch_id, L/log_period,scheduler.get_lr() ,cur_train_period_acc))
                R=S=L=0
                train_acc=max(cur_train_period_acc,train_acc)
            if (batch_id) % test_period == 0 and train_acc>=test_acc:
                fast_infer.load_dict(model.state_dict())
                fast_infer._init_fuse_params()
                fast_infer.eval()
                with paddle.no_grad():
                    TR=0
                    for data in test_loader():
                        predicts = fast_infer(data['img']).transpose([1,2,0])[:,0,:]
                        test_right,_ = metric(predicts,data['label'],tokenizer)
                        TR+=test_right
                cur_test_acc=TR/len(test_dataset)
                print("epoch: {}, batch_id: {}, test_acc is: {}".format(epoch, batch_id, cur_test_acc))
                model.train()
                if cur_test_acc>test_acc:
                    test_acc = cur_test_acc
                    train_acc=1
                    print("save model's params to ./best_model.pdparams")
                    paddle.save(model.state_dict(),"./best_model.pdparams")
                        
if __name__ == '__main__':
    args = parse_args()
    train(args)