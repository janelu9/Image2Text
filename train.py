#!/usr/bin/env python
# coding: utf-8
# Created on Mon Apr 11 16:55:39 2022
# @author: Lu Jian
# Email:janelu@live.cn;

from data_loader import SimpleDataSet
from image_aug import image_process
import paddle
import paddle.nn as nn
from paddle.io import DataLoader,DistributedBatchSampler
from paddlenlp.transformers import GPTChineseTokenizer
import paddle.distributed as dist
import os
from image2text import (SwinTransformerEncoder,TransformerDecoder,Image2Text,WordEmbedding,
                        PositionalEmbedding,FasterTransformer,InferTransformerModel)                        
from cswin_transformer import CSwinTransformerEncoder
from lr_scheduler import InverseSqrt,
from argparse import ArgumentParser
import numpy as np
from time import time

def parse_args():

    parser = ArgumentParser(description="Paddle distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    parser.add_argument("--decoder-pretrained", type=str,default="ernie_3.0_medium_zh.pdparams",
                        help="pretrained model's params path")
    #https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh.pdparams
    
    parser.add_argument("--data-dir", type=str,default="/mnt/e/OCR/unilm/trocr/IAM/image/",
                        help="data _dir")
    
    parser.add_argument("--train-list", type=str,default="/mnt/e/OCR/unilm/trocr/IAM/gt_test.txt",
                        help="train data dir")
                             
    parser.add_argument("--test-list", type=str,default="/mnt/e/OCR/unilm/trocr/IAM/gt_test.txt",
                        help="test data dir")                         
    return parser.parse_args()

class ocr_token:
    def __init__(self,keys_path):
        super().__init__()
        with open(keys_path) as f :keys=f.read().splitlines()
        self.string2id={"":0,"BOS":1,"EOS":2}
        self.string2id.update({k:i+3 for i,k in enumerate(keys)})
        self.id2string={i:k for k,i in self.string2id.items()}
        self.bos_token_id = self.string2id["BOS"]
        self.eos_token_id = self.string2id["EOS"]
        self.vocab_size=len(self.string2id)
        
    def __call__(self,text):
        return {"input_ids":[self.string2id.get(i,"") for i in text]}
    
    def convert_ids_to_string(self,ids):
        return "".join([self.id2string[i] for i in ids])

def train(args):
    dist.init_parallel_env()
    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id)
    device = paddle.set_device(device)    
    tokenizer = ocr_token("ocr_keys_v1.txt") #GPTChineseTokenizer("gpt-cpm-cn-sentencepiece.model")
    train_dataset=SimpleDataSet(args.data_dir,args.train_list,image_process(224),tokenizer)
    test_dataset=SimpleDataSet(args.data_dir,args.test_list,image_process(224,False),tokenizer)
    sampler = DistributedBatchSampler(train_dataset, batch_size=32,shuffle=True,drop_last=False)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler =sampler,
        places=device,
        num_workers=4,
        return_list=True,
        collate_fn = train_dataset.collate_fn,
        use_shared_memory=True)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False, 
        batch_size=16,
        drop_last=False,
        places=device,
        num_workers=2,
        return_list=True,
        collate_fn = test_dataset.collate_fn,
        use_shared_memory=True)
        
    #encoder = SwinTransformerEncoder(img_size=224,embed_dim=96,depths=[2, 2, 18, 2],num_heads=[3, 6, 12, 24],window_size=7,drop_path_rate=0.1)
    encoder= CSwinTransformerEncoder(
        image_size=224,
        embed_dim=96,
        depths=[2, 4, 32, 2],
        splits=[1, 2, 7, 7],
        num_heads=[4, 8, 16, 32],
        droppath=0.5,
        )
    encoder.load_dict(paddle.load("CSWinTransformer_base_224_pretrained.pdparams"))
    #https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSWinTransformer_base_224_pretrained.pdparams
    decoder = TransformerDecoder(d_model=768,n_head=12,dim_feedforward=768*4,num_layers=6,dropout=0.1)
    word_emb = WordEmbedding(vocab_size=tokenizer.vocab_size,emb_dim=decoder.d_model,pad_id=train_dataset.pad_id)
    pos_emb = PositionalEmbedding(decoder.d_model,max_length=2048,learned=True)
    project_out = nn.Linear(decoder.d_model, word_emb.vocab_size)
    
    def load_pretrained_gpt(path):
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
        
    def load_pretrained_ernie(param_path,vocab_path,token):
        p= paddle.load(param_path)
        with open(vocab_path) as f :keys=f.read().splitlines()
        other_string2id = {k:i for i,k in enumerate(keys)}
        index = [ other_string2id["[UNK]"], other_string2id["[PAD]"], other_string2id['[SEP]']]
        index.extend([other_string2id.get(token.id2string[i].lower(),other_string2id["[UNK]"]) for i in range(3,token.vocab_size)])
        word_emb.load_dict({
            'word_embeddings.weight': paddle.index_select(p['ernie.embeddings.word_embeddings.weight'],paddle.to_tensor(index)),
            'layer_norm.weight' : p['ernie.embeddings.layer_norm.weight'], 
            'layer_norm.bias' :p['ernie.embeddings.layer_norm.bias']        
                           })
        pos_emb.load_dict({'position_embeddings.weight':p['ernie.embeddings.position_embeddings.weight']})
        
        decoder_state_dict={}
        un_matched = []
        for k in decoder.state_dict().keys():
            ek = "ernie.encoder."+k
            if ek in p:
                decoder_state_dict[k]=p[ek]
            else:
                un_matched.append(k)
        decoder.load_dict(decoder_state_dict)
        if len(un_matched)>0:
            print("unmatched keys:%s" % str(un_matched))
    try:
        #load_pretrained_gpt(args.decoder_pretrained)
        load_pretrained_ernie(args.decoder_pretrained,"ernie_vocab.txt",tokenizer)
    except:
        print("pretrained decoder isn't loaded")
    model=Image2Text(encoder,decoder,word_emb,pos_emb,project_out,train_dataset.eos_id)
    try:
        model.load_dict(paddle.load("./check_point.pdparams"))
        print("load model's params from last epoch")
    except:
        print("start trainning from zero")
    try:
        fast = True
        infer = FasterTransformer(model,max_out_len=32)
    except:
        fast = False
        infer = InferTransformerModel(model,max_out_len=32,beam_search_version="custom")    
    model = paddle.DataParallel(model)
    model.train()
    
    epochs=50
    batch_id=0
    # scheduler = paddle.optimizer.lr.NoamDecay(d_model=decoder.d_model, warmup_steps=500, verbose=False)
    # opt = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=scheduler,weight_decay= 0.0001)
    scheduler = LinearDecayWithWarmup(2e-5, epochs*len(train_loader),5000,last_epoch = batch_id-1)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    opt = paddle.optimizer.AdamW(
        learning_rate=scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-6,
        parameters=model.parameters(),
        weight_decay=0.0,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=nn.ClipGradByGlobalNorm(1.))
    
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
        

    R=S=L=0
    log_period=200
    test_period=1000
    train_acc=0
    test_acc=0.65
    st=time()
    for epoch in range(epochs):    
        for data in train_loader():
            predicts = model(data['img'],data['tgt'],tgt_mask=True)
            loss = paddle.nn.functional.cross_entropy(predicts, data['label'])
            right,samples = metric(predicts,data['label'],tokenizer)
            loss.backward()
            opt.step()
            opt.clear_grad()
            scheduler.step()
            
            batch_id+=1
            R += right
            S += samples
            L += loss.numpy()[0]
            if (batch_id) % log_period == 0:
                cur_train_period_acc = R/S
                print("epoch: {}, batch_id: {}, loss: {:.6f}, lr: {:.9f}, acc: {:.6f}, fps:{:.6f}".\
                format(epoch, batch_id, L/log_period,scheduler.get_lr() ,cur_train_period_acc,S/(time()-st)))
                R=S=L=0
                train_acc=max(cur_train_period_acc,train_acc)
                st=time()
            if (batch_id) % test_period == 0 and train_acc>=test_acc:
                paddle.save(model.state_dict(),"./check_point.pdparams")
                print(f"save model's params to ./check_point.pdparams with batch_id :{batch_id}")
                if fast:
                    infer._init_fuse_params()
                infer.eval()
                with paddle.no_grad():
                    TR=0
                    for data in test_loader():
                        predicts = infer(data['img'])
                        if fast:
                            predicts=predicts.transpose([1,0,2])
                        predicts=predicts[:,:,0]
                        test_right,_ = metric(predicts,data['label'],tokenizer)
                        TR+=test_right
                cur_test_acc=TR/len(test_dataset)
                print("epoch: {}, batch_id: {}, test_acc : {:.6f}, cost_time: {:.6f}s".format(epoch, batch_id, cur_test_acc,(time()-st)))
                model.train()
                if cur_test_acc>test_acc:
                    test_acc = cur_test_acc
                    train_acc=1
                    paddle.save(model.state_dict(),"./best_model.pdparams")
                    print(f"save model's params to ./best_model.pdparams with batch_id :{batch_id} acc :{test_acc}")
                st=time()     
if __name__ == '__main__':
    args = parse_args()
    train(args)