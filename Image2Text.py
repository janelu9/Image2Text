#!/usr/bin/env python
# coding: utf-8
# Created on Fri Apr 1 09:36:49 2022
# @author: Lu Jian
# Email:janelu@live.cn; lujian@sdc.icbc.com.cn

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from vision_transformer import VisionTransformer, Identity, trunc_normal_, zeros_
from swin_transformer import SwinTransformer
from paddle.framework import ParamAttr
from paddle.nn.layer.transformer import _convert_param_attr_to_list
import collections
from paddlenlp.ops import InferTransformerDecoding
from paddlenlp.transformers import TransformerBeamSearchDecoder

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self,**kwargs):
        super().__init__(class_num=0,**kwargs)
        self.pos_embed = self.create_parameter(
            shape=(1, self.patch_embed.num_patches + 2, self.embed_dim), default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)

        self.dist_token = self.create_parameter(
            shape=(1, 1, self.embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)

        trunc_normal_(self.dist_token)
        trunc_normal_(self.pos_embed)
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand((B, -1, -1))
        dist_token = self.dist_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, dist_token, x), axis=1)
        
        input_embedding = x + self.pos_embed
        x = self.pos_drop(input_embedding)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        return x

class SwinTransformerEncoder(SwinTransformer):
    def __init__(self,**kwargs):
        super().__init__(class_num=0,**kwargs)
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        return x
    def forward(self,x):
        x = self.forward_features(x)
        return x
    
class PositionalEmbedding(nn.Layer):
    def __init__(self, emb_dim, max_length,learned = False):
        super(PositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.max_length = max_length
        self.position_embeddings = nn.Embedding(num_embeddings=max_length,embedding_dim=self.emb_dim,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(0., emb_dim**-0.5)))
        if not learned:
            w = paddle.zeros((max_length, emb_dim),paddle.float32)
            pos = paddle.arange(0, max_length, dtype=paddle.float32).unsqueeze(1)
            div = (-paddle.arange(0, emb_dim, 2,dtype=paddle.float32)/emb_dim * paddle.to_tensor(10000,paddle.float32).log()).exp()
            w[:, 0::2] = paddle.sin(pos * div)
            w[:, 1::2] = paddle.cos(pos * div)
            self.position_embeddings.weight.set_value(w)
            self.position_embeddings.weight.stop_gradient = True
            
    def forward(self, pos):
        return self.position_embeddings(pos)
    
class WordEmbedding(nn.Layer):
    def __init__(self, vocab_size, emb_dim, pad_id=0):
        super(WordEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim,padding_idx=pad_id,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(0., emb_dim**-0.5)))
    def forward(self, word):
        return self.emb_dim**0.5 * self.word_embeddings(word)
 
        
class MultiHeadAttention(nn.Layer):
    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])
    def __init__(self,embed_dim,num_heads,dropout=0.,kdim=None,vdim=None,need_weights=False,weight_attr=None,bias_attr=None,**kwargs):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = nn.Linear(self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = nn.Linear(self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = nn.Linear(embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value, cache=None):
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        if isinstance(cache, self.StaticCache):
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)
        if isinstance(cache, self.Cache):
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
            cache = self.Cache(k, v)
        return (q, k, v) if cache is None else (q, k, v, cache)
    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  #
            k = tensor.zeros([key.shape[0], self.num_heads, 0, self.head_dim],dtype=key.dtype)
            v = tensor.zeros([key.shape[0], self.num_heads, 0, self.head_dim],dtype=key.dtype)
            return self.Cache(k, v)
        else:
            return self.Cache(key, value)

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)
        product = paddle.matmul(
            x=q * (self.head_dim**-0.5), y=k, transpose_y=True)
        if attn_mask is not None:
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")
        out = tensor.matmul(weights, v)
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        out = self.out_proj(out)
        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)

class TransformerDecoderLayer(nn.Layer):
    def __init__(self,d_model,nhead,dim_feedforward,dropout=0.0,skdim=None,svdim=None,ckdim=None,cvdim=None,activation='ReLU',
                 attn_dropout=None,act_dropout=None,normalize_before=True,weight_attr=None,bias_attr=None,**kwargs):
        self._config = locals()
        self._config.pop("__class__", None)
        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before
        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model,nhead,dropout=attn_dropout,kdim=skdim,vdim=svdim
                                            ,weight_attr=weight_attrs[0],bias_attr=bias_attrs[0],**kwargs)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model,nhead,dropout=attn_dropout,kdim=ckdim,vdim=cvdim,
                                             weight_attr=weight_attrs[1],bias_attr=bias_attrs[1],**kwargs)
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        self.activation = eval(f'nn.{activation}()')#getattr(F, activation)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])
        self.dropout3 = nn.Dropout(dropout, mode="upscale_in_train")
        self._config.pop("self")

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, cache[0])
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if cache is None:
            tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else:
            tgt, static_cache = self.cross_attn(tgt, memory, memory, memory_mask, cache[1])
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache,static_cache))

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        static_cache = self.cross_attn.gen_cache(memory, memory, type=self.cross_attn.StaticCache)
        return incremental_cache, static_cache
    
class TransformerDecoder(nn.Layer):
    def __init__(self,d_model, n_head, dim_feedforward, num_layers, **kwargs):
        super(TransformerDecoder, self).__init__()
        decoder_layer = TransformerDecoderLayer(d_model,n_head,dim_feedforward,**kwargs)
        self.layers = nn.LayerList([(decoder_layer if i == 0 else
                                  type(decoder_layer)(**decoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.n_head= n_head
        self.d_model =d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        output = tgt 
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output,
                             memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             cache=None)
            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)
                
        if self.norm is not None:
             output = self.norm(output)
            
        return output if cache is None else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache
    
    def _mask(self,length):
        return tensor.triu((paddle.zeros((length, length), dtype=paddle.get_default_dtype()) -float('inf')),1)
    
class Image2Text(nn.Layer):
    def __init__(self,img_encoder,txt_decoder,word_emb,pos_emb,project_out,eos_id=7,dropout=0):
        super(Image2Text, self).__init__()
        self.encoder = img_encoder
        self.decoder = txt_decoder
        self.vocab_size = word_emb.vocab_size
        self.bos_id = word_emb.pad_id
        self.eos_id = eos_id
        self.max_length = pos_emb.max_length
        self.word_embedding = word_emb
        self.pos_embedding = pos_emb
        self.dropout= nn.Dropout(dropout)
        self.project_out = project_out

    def forward(self, img, tgt,src_mask=None,tgt_mask=None, memory_mask=None):         
        memory = self.encoder(img)            
        dec_input = self.dropout(self.word_embedding(tgt) + \
                                 self.pos_embedding(paddle.arange(tgt.shape[1]).unsqueeze(0)))
        tgt_mask= self.decoder._mask(tgt.shape[1]) if tgt_mask is not None else None            
        dec_output = self.decoder(dec_input,memory,tgt_mask=tgt_mask)
        predict = self.project_out(dec_output)
        return predict
    
class FasterTransformer(nn.Layer):
    def __init__(self,model,
                 decoding_strategy="beam_search",
                 beam_size=4,
                 topk=4,
                 topp=0.0,
                 max_out_len=256,
                 diversity_rate=0.0,
                 decoding_lib=None,
                 use_fp16_decoding=False,
                 enable_faster_encoder=False,
                 use_fp16_encoder=False,
                 rel_len=False,
                 alpha=0.6):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.decoding_strategy = args.pop("decoding_strategy")
        self.beam_size = args.pop("beam_size")
        self.topk = args.pop("topk")
        self.topp = args.pop("topp")
        self.max_out_len = args.pop("max_out_len")
        self.diversity_rate = args.pop("diversity_rate")
        self.decoding_lib = args.pop("decoding_lib")
        self.use_fp16_decoding = args.pop("use_fp16_decoding")
        self.enable_faster_encoder = args.pop("enable_faster_encoder")
        self.use_fp16_encoder = args.pop("use_fp16_encoder")
        self.rel_len = args.pop("rel_len")
        self.alpha = args.pop("alpha")
        super(FasterTransformer, self).__init__()
        
        self.encoder=model.encoder
        
        self.decoding = InferTransformerDecoding(
            decoder=model.decoder,
            word_embedding=model.word_embedding.word_embeddings,
            positional_embedding=model.pos_embedding.position_embeddings,
            linear=model.project_out,
            num_decoder_layers=model.decoder.num_layers,
            n_head=model.decoder.n_head,
            d_model=model.decoder.d_model,
            bos_id=model.bos_id,
            eos_id=model.eos_id,
            decoding_strategy=decoding_strategy,
            beam_size=beam_size,
            topk=topk,
            topp=topp,
            max_out_len=max_out_len,
            diversity_rate=self.diversity_rate,
            decoding_lib=self.decoding_lib,
            use_fp16_decoding=self.use_fp16_decoding,
            rel_len=self.rel_len,
            alpha=self.alpha)
        
        if self.decoding._fuse_qkv:
            self._init_fuse_params(model.decoder.state_dict())
            
    def _init_fuse_params(self,decoder_state_dict):
        fuse_param={}
        for item in self.decoding.state_dict().keys():
            _, param_type ,num_layer = item.rsplit("_",2)
            fuse_param[item]= paddle.concat((decoder_state_dict["layers.%s.self_attn.q_proj.%s" % (num_layer,param_type)],
                                             decoder_state_dict["layers.%s.self_attn.k_proj.%s" % (num_layer,param_type)],
                                             decoder_state_dict["layers.%s.self_attn.v_proj.%s" % (num_layer,param_type)],
                                             ),-1)
        self.decoding.load_dict(fuse_param)
                 
            
    def forward(self, img, trg_word=None):
        enc_output = self.encoder(img)
        if self.use_fp16_decoding and enc_output.dtype != paddle.float16:
            enc_output = paddle.cast(enc_output, dtype="float16")
        elif not self.use_fp16_decoding and enc_output.dtype != paddle.float32:
            enc_output = paddle.cast(enc_output, dtype="float32")
        mem_seq_lens = paddle.ones([enc_output.shape[0]],paddle.int32)*enc_output.shape[1]
        ids = self.decoding(enc_output, mem_seq_lens, trg_word=trg_word)
        return ids


class TransformerDecodeCell(nn.Layer):
    def __init__(self,
                 decoder,
                 word_embedding=None,
                 pos_embedding=None,
                 linear=None,
                 dropout=None):
        super(TransformerDecodeCell, self).__init__()
        self.decoder = decoder
        self.word_embedding = word_embedding
        self.pos_embedding = pos_embedding
        self.linear = linear
        self.dropout =dropout

    def forward(self, inputs, states, static_cache, trg_src_attn_bias, memory,
                **kwargs):
  
        if states and static_cache:
            states = list(zip(states, static_cache))

        if self.word_embedding:
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs)

            word_emb = self.word_embedding(inputs[0])
            pos_emb = self.pos_embedding(inputs[1])
            word_emb = word_emb + pos_emb
            inputs = self.dropout(word_emb)

            cell_outputs, new_states = self.decoder(inputs, memory, None,
                                                    trg_src_attn_bias, states)
        else:
            cell_outputs, new_states = self.decoder(inputs, memory, None,
                                                    trg_src_attn_bias, states)

        if self.linear:
            cell_outputs = self.linear(cell_outputs)

        new_states = [cache[0] for cache in new_states]

        return cell_outputs, new_states
        
class InferTransformerModel(nn.Layer):
    def __init__(self,model,
                 beam_size=4,
                 max_out_len=256,
                 output_time_major=False,
                 beam_search_version='v1',
                 **kwargs):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.beam_size = args.pop("beam_size")
        self.max_out_len = args.pop("max_out_len")
        self.output_time_major = args.pop("output_time_major")
        self.beam_search_version = args.pop('beam_search_version')
        kwargs = args.pop("kwargs")
        if self.beam_search_version == 'v2':
            self.alpha = kwargs.get("alpha", 0.6)
            self.rel_len = kwargs.get("rel_len", False)
        super(InferTransformerModel, self).__init__()

        cell = TransformerDecodeCell(
            model.decoder, model.word_embedding,
            model.pos_embedding, model.project_out, model.dropout)

        self.decode = TransformerBeamSearchDecoder(
            cell, model.bos_id, model.eos_id, beam_size, var_dim_in_state=2)
            
        self.model=model

    def forward(self, enc_input, trg_word=None):
    
        if trg_word is not None:
            trg_length = paddle.sum(paddle.cast(
                trg_word != self.bos_id, dtype="int32"),
                                    axis=-1)
        else:
            trg_length = None

        if self.beam_search_version == 'v1':

            enc_output = self.model.encoder(enc_input)

            # Init states (caches) for transformer, need to be updated according to selected beam
            incremental_cache, static_cache = self.model.decoder.gen_cache(
                enc_output, do_zip=True)

            static_cache, enc_output = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                (static_cache, enc_output), self.beam_size)

            rs, _ = nn.decode.dynamic_decode(
                decoder=self.decode,
                inits=incremental_cache,
                max_step_num=self.max_out_len,
                memory=enc_output,
                trg_src_attn_bias=None,
                static_cache=static_cache,
                is_test=True,
                output_time_major=self.output_time_major,
                trg_word=trg_word,
                trg_length=trg_length)

            return rs

        elif self.beam_search_version == 'v2':
            finished_seq, finished_scores = self.beam_search_v2(
                enc_input, self.beam_size, self.max_out_len, self.alpha,
                trg_word, trg_length)
            if self.output_time_major:
                finished_seq = finished_seq.transpose([2, 0, 1])
            else:
                finished_seq = finished_seq.transpose([0, 2, 1])

            return finished_seq,finished_scores

    def beam_search_v2(self,
                       enc_input,
                       beam_size=4,
                       max_len=None,
                       alpha=0.6,
                       trg_word=None,
                       trg_length=None):


        def expand_to_beam_size(tensor, beam_size):
            tensor = paddle.unsqueeze(tensor, axis=1)
            tile_dims = [1] * len(tensor.shape)
            tile_dims[1] = beam_size
            return paddle.tile(tensor, tile_dims)

        def merge_beam_dim(tensor):
            shape = tensor.shape
            return paddle.reshape(tensor,
                                  [shape[0] * shape[1]] + list(shape[2:]))

        enc_output = self.model.encoder(enc_input)

        # constant number
        inf = float(1. * 1e7)
        batch_size = enc_output.shape[0]
        max_len = (enc_output.shape[1] + 20) if max_len is None else (
            enc_output.shape[1] + max_len if self.rel_len else max_len)

        ### initialize states of beam search ###
        ## init for the alive ##
        initial_log_probs = paddle.assign(
            np.array(
                [[0.] + [-inf] * (beam_size - 1)], dtype="float32"))
        alive_log_probs = paddle.tile(initial_log_probs, [batch_size, 1])

        alive_seq = paddle.tile(
            paddle.cast(
                paddle.assign(np.array([[[self.model.bos_id]]])), "int64"),
            [batch_size, beam_size, 1])

        ## init for the finished ##
        finished_scores = paddle.assign(
            np.array(
                [[-inf] * beam_size], dtype="float32"))
        finished_scores = paddle.tile(finished_scores, [batch_size, 1])

        finished_seq = paddle.tile(
            paddle.cast(
                paddle.assign(np.array([[[self.model.bos_id]]])), "int64"),
            [batch_size, beam_size, 1])
        finished_flags = paddle.zeros_like(finished_scores)

        ### initialize inputs and states of transformer decoder ###
        ## init inputs for decoder, shaped `[batch_size*beam_size, ...]`
        pre_word = paddle.reshape(alive_seq[:, :, -1],
                                  [batch_size * beam_size, 1])
        trg_src_attn_bias = None

        enc_output = merge_beam_dim(expand_to_beam_size(enc_output, beam_size))

        ## init states (caches) for transformer, need to be updated according to selected beam
        caches = self.model.decoder.gen_cache(enc_output, do_zip=False)

        if trg_word is not None:
            scores_dtype = finished_scores.dtype
            scores = paddle.ones(
                shape=[batch_size, beam_size * 2], dtype=scores_dtype) * -1e4
            scores = paddle.scatter(
                scores.flatten(),
                paddle.arange(
                    0,
                    batch_size * beam_size * 2,
                    step=beam_size * 2,
                    dtype=finished_seq.dtype),
                paddle.zeros([batch_size]))
            scores = paddle.reshape(scores, [batch_size, beam_size * 2])

        def update_states(caches, topk_coordinates, beam_size, batch_size):
            new_caches = []
            for cache in caches:
                k = gather_2d(
                    cache[0].k,
                    topk_coordinates,
                    beam_size,
                    batch_size,
                    need_unmerge=True)
                v = gather_2d(
                    cache[0].v,
                    topk_coordinates,
                    beam_size,
                    batch_size,
                    need_unmerge=True)
                new_caches.append((MultiHeadAttention.Cache(k, v), cache[1]))
            return new_caches

        def get_topk_coordinates(beam_idx, beam_size, batch_size,
                                 dtype="int64"):
            batch_pos = paddle.arange(
                batch_size * beam_size, dtype=dtype) // beam_size
            batch_pos = paddle.reshape(batch_pos, [batch_size, beam_size])
            topk_coordinates = paddle.stack([batch_pos, beam_idx], axis=2)
            return topk_coordinates

        def gather_2d(tensor_nd,
                      topk_coordinates,
                      beam_size,
                      batch_size,
                      need_unmerge=False):

            new_tensor_nd = paddle.reshape(
                tensor_nd,
                shape=[batch_size, beam_size] +
                list(tensor_nd.shape[1:])) if need_unmerge else tensor_nd
            topk_seq = paddle.gather_nd(new_tensor_nd, topk_coordinates)
            return merge_beam_dim(topk_seq) if need_unmerge else topk_seq

        def early_finish(alive_log_probs, finished_scores,
                         finished_in_finished):
            max_length_penalty = np.power(((5. + max_len) / 6.), alpha)
            lower_bound_alive_scores = alive_log_probs[:,
                                                       0] / max_length_penalty
            lowest_score_of_fininshed_in_finished = paddle.min(
                finished_scores * finished_in_finished, 1)
            lowest_score_of_fininshed_in_finished += (
                1. - paddle.max(finished_in_finished, 1)) * -inf
            bound_is_met = paddle.all(
                paddle.greater_than(lowest_score_of_fininshed_in_finished,
                                    lower_bound_alive_scores))

            return bound_is_met

        def grow_topk(i, logits, alive_seq, alive_log_probs, states):
            """
            This function takes the current alive sequences, and grows them to topk
            sequences where k = 2*beam.
            """
            logits = paddle.reshape(logits, [batch_size, beam_size, -1])
            candidate_log_probs = paddle.log(F.softmax(logits, axis=2))
            log_probs = paddle.add(candidate_log_probs,
                                   alive_log_probs.unsqueeze(-1))

            # Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
            # https://arxiv.org/abs/1609.08144.
            length_penalty = paddle.pow((5.0 + i + 1.0) / 6.0, alpha)
            curr_scores = log_probs / length_penalty
            flat_curr_scores = paddle.reshape(curr_scores, [batch_size, -1])

            topk_scores, topk_ids = paddle.topk(
                flat_curr_scores, k=beam_size * 2)
            if topk_ids.dtype != alive_seq.dtype:
                topk_ids = paddle.cast(topk_ids, dtype=alive_seq.dtype)

            if trg_word is not None:
                topk_ids, topk_scores = force_decoding_v2(topk_ids, topk_scores,
                                                          i)

            topk_log_probs = topk_scores * length_penalty

            topk_beam_index = topk_ids // self.model.vocab_size
            topk_ids = topk_ids % self.model.vocab_size

            topk_coordinates = get_topk_coordinates(
                topk_beam_index,
                beam_size * 2,
                batch_size,
                dtype=alive_seq.dtype)
            topk_seq = gather_2d(alive_seq, topk_coordinates, beam_size,
                                 batch_size)
            topk_seq = paddle.concat(
                [
                    topk_seq, paddle.reshape(topk_ids,
                                             list(topk_ids.shape[:]) + [1])
                ],
                axis=2)
            states = update_states(states, topk_coordinates, beam_size,
                                   batch_size)
            eos = paddle.full(
                shape=paddle.shape(topk_ids),
                dtype=alive_seq.dtype,
                fill_value=self.model.eos_id)
            topk_finished = paddle.cast(paddle.equal(topk_ids, eos), "float32")

            # topk_seq: [batch_size, 2*beam_size, i+1]
            # topk_log_probs, topk_scores, topk_finished: [batch_size, 2*beam_size]
            return topk_seq, topk_log_probs, topk_scores, topk_finished, states

        def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished,
                       states):
            curr_scores += curr_finished * -inf
            _, topk_indexes = paddle.topk(curr_scores, k=beam_size)
            if topk_indexes.dtype != curr_seq.dtype:
                topk_indexes = paddle.cast(topk_indexes, dtype=curr_seq.dtype)

            topk_coordinates = get_topk_coordinates(
                topk_indexes, beam_size, batch_size, dtype=curr_seq.dtype)
            alive_seq = gather_2d(curr_seq, topk_coordinates, beam_size,
                                  batch_size)

            alive_log_probs = gather_2d(curr_log_probs, topk_coordinates,
                                        beam_size, batch_size)
            states = update_states(states, topk_coordinates, beam_size * 2,
                                   batch_size)

            return alive_seq, alive_log_probs, states

        def grow_finished(finished_seq, finished_scores, finished_flags,
                          curr_seq, curr_scores, curr_finished):
            # finished scores
            finished_seq = paddle.concat(
                [
                    finished_seq, paddle.full(
                        shape=[batch_size, beam_size, 1],
                        dtype=finished_seq.dtype,
                        fill_value=self.model.eos_id)
                ],
                axis=2)
            curr_scores += (1. - curr_finished) * -inf
            curr_finished_seq = paddle.concat([finished_seq, curr_seq], axis=1)
            curr_finished_scores = paddle.concat(
                [finished_scores, curr_scores], axis=1)
            curr_finished_flags = paddle.concat(
                [finished_flags, curr_finished], axis=1)
            _, topk_indexes = paddle.topk(curr_finished_scores, k=beam_size)
            if topk_indexes.dtype != curr_seq.dtype:
                topk_indexes = paddle.cast(topk_indexes, dtype=curr_seq.dtype)

            topk_coordinates = get_topk_coordinates(
                topk_indexes, beam_size, batch_size, dtype=curr_seq.dtype)
            finished_seq = gather_2d(curr_finished_seq, topk_coordinates,
                                     beam_size, batch_size)
            finished_scores = gather_2d(curr_finished_scores, topk_coordinates,
                                        beam_size, batch_size)
            finished_flags = gather_2d(curr_finished_flags, topk_coordinates,
                                       beam_size, batch_size)

            return finished_seq, finished_scores, finished_flags

        def force_decoding_v2(topk_ids, topk_scores, time):
            beam_size = topk_ids.shape[1]
            if trg_word.shape[1] > time:
                force_position = paddle.unsqueeze(trg_length > time, [1])
                force_position.stop_gradient = True
                force_position = paddle.tile(force_position, [1, beam_size])

                crt_trg_word = paddle.slice(
                    trg_word, axes=[1], starts=[time], ends=[time + 1])
                crt_trg_word = paddle.tile(crt_trg_word, [1, beam_size])

                topk_ids = paddle.where(force_position, crt_trg_word, topk_ids)

                topk_scores = paddle.where(force_position, scores, topk_scores)

            return topk_ids, topk_scores

        def inner_loop(i, pre_word, alive_seq, alive_log_probs, finished_seq,
                       finished_scores, finished_flags, caches):
            trg_pos = paddle.full(
                shape=paddle.shape(pre_word),
                dtype=alive_seq.dtype,
                fill_value=i)
            trg_emb = self.model.word_embedding(pre_word)
            trg_pos_emb = self.model.pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = self.model.dropout(trg_emb)

            logits, caches = self.model.decoder(
                dec_input, enc_output, None, trg_src_attn_bias, caches)
            logits = self.model.project_out(logits)
            topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
                i, logits, alive_seq, alive_log_probs, caches)
            alive_seq, alive_log_probs, states = grow_alive(
                topk_seq, topk_scores, topk_log_probs, topk_finished, states)
            caches = states
            finished_seq, finished_scores, finished_flags = grow_finished(
                finished_seq, finished_scores, finished_flags, topk_seq,
                topk_scores, topk_finished)
            pre_word = paddle.reshape(alive_seq[:, :, -1],
                                      [batch_size * beam_size, 1])
            return (i + 1, pre_word, alive_seq, alive_log_probs, finished_seq,
                    finished_scores, finished_flags, caches)

        def is_not_finish(i, pre_word, alive_seq, alive_log_probs, finished_seq,
                          finished_scores, finished_flags, caches):
            return paddle.greater_than(
                i < max_len,
                early_finish(alive_log_probs, finished_scores, finished_flags))

        _, pre_word, alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags, caches = paddle.static.nn.while_loop(
            is_not_finish,
            inner_loop, [
                paddle.zeros(
                    shape=[1],
                    dtype="int64"), pre_word, alive_seq, alive_log_probs,
                finished_seq, finished_scores, finished_flags, caches
            ],
            is_test=True
        )
        finished_flags = paddle.cast(finished_flags, dtype=finished_seq.dtype)
        neg_finished_flags = 1 - finished_flags
        finished_seq = paddle.multiply(
            finished_seq, finished_flags.unsqueeze(-1)) + paddle.multiply(
                alive_seq, neg_finished_flags.unsqueeze(-1))
        finished_scores = paddle.multiply(
            finished_scores,
            paddle.cast(
                finished_flags, dtype=finished_scores.dtype)) + paddle.multiply(
                    alive_log_probs,
                    paddle.cast(
                        neg_finished_flags, dtype=alive_log_probs.dtype))
        return finished_seq, finished_scores

encoder = SwinTransformerEncoder(embed_dim=48,depths=[2, 2, 6, 2],num_heads=[3, 6, 12, 24],window_size=7,drop_path_rate=0.2)
decoder = TransformerDecoder(d_model=384,n_head=6,dim_feedforward=1536,num_layers=6)
word_emb = WordEmbedding(vocab_size=64044,emb_dim=decoder.d_model,pad_id=0)
pos_emb = PositionalEmbedding(decoder.d_model,max_length=512)
project_out = nn.Linear(decoder.d_model, word_emb.vocab_size)

model=Image2Text(encoder,decoder,word_emb,pos_emb,project_out)

fast_infer = FasterTransformer(model,max_out_len=20)
infer = InferTransformerModel(model,max_out_len=20)

img = paddle.rand((1,3,224,224))
tgt = paddle.randint(shape=(1,20),low=1,high=64044)

model(img,tgt)

fast_infer.eval()
with paddle.no_grad():
    out=fast_infer(img)
    
infer.eval()
with paddle.no_grad():
    out1=infer(img)
