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
from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser(description="Paddle distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    parser.add_argument("--decoder-pretrained", type=str,default="../gpt/gpt-cpm-small-cn-distill.pdparams",
                        help="pretrained model's params path")
    
    parser.add_argument("--data_dir", type=str,default="/mnt/e/OCR/unilm/trocr/IAM/image/",
                        help="data _dir")
    
    parser.add_argument("--train_list", type=str,default="/mnt/e/OCR/unilm/trocr/IAM/gt_test.txt",
                        help="data _dir")
                             
    return parser.parse_args()


def train(args):
    dist.init_parallel_env()
    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id)
    device = paddle.set_device(device)
    tokenizer = GPTChineseTokenizer("gpt-cpm-cn-sentencepiece.model")
    dataset=SimpleDataSet(args.data_dir,args.train_list,image_process(224),tokenizer)

    train_loader = DataLoader(
        dataset=dataset,
        shuffle=True, 
        batch_size=8,
        drop_last=False,
        places=device,
        num_workers=1,
        return_list=True,
        collate_fn = dataset.collate_fn,
        use_shared_memory=True)
        
    encoder = SwinTransformerEncoder(img_size=224,embed_dim=48,depths=[2, 2, 6, 2],num_heads=[3, 6, 12, 24],window_size=7,drop_path_rate=0.2)
    decoder = TransformerDecoder(d_model=384,n_head=6,dim_feedforward=1536,num_layers=6)
    word_emb = WordEmbedding(vocab_size=tokenizer.vocab_size,emb_dim=decoder.d_model,pad_id=dataset.pad_id)
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
    model=Image2Text(encoder,decoder,word_emb,pos_emb,project_out,dataset.eos_id)
    fast_infer = FasterTransformer(model,max_out_len=70)
    model = paddle.DataParallel(model)
    model.train()
    scheduler = paddle.optimizer.lr.NoamDecay(d_model=decoder.d_model, warmup_steps=500, verbose=False)
    adam = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=scheduler,weight_decay= 0.0001)
    epochs=5
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            predicts = model(data['img'],data['tgt'],True) 
            loss = paddle.nn.functional.cross_entropy(predicts, data['label'])
            acc = 1
            '''
            train metric codes
            '''
            loss.backward()
            if (batch_id+1) % 10 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc))
            adam.step()
            adam.clear_grad()
            scheduler.step()
        '''
        eval test dataset codes
        save model's params codes
        '''
if __name__ == '__main__':
    args = parse_args()
    train(args)