# Image2Text Model
An image to text model base on transformer which can also be used on OCR task.

在这里集成了 NVIDIA [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1) 用于预测加速。同时集成了 FasterTransformer float32 以及 float16 预测。以下是使用 FasterTransformer 的说明。

## 环境说明

* 本项目依赖于 PaddlePaddle 2.1.0 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1 或以上版本 （需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup) 使用必要的环境
* 环境依赖
  - attrdict
  - pyyaml
  - paddlenlp
  ```shell
  pip install attrdict pyyaml paddlenlp
  ```
* 对于图灵架构GPU(CUDA版本为10.1)也可以通过docker镜像来安装paddle运行环境

` docker pull registry.cn-shanghai.aliyuncs.com/janelu9/dl:paddle2.2.2-cuda10.1-cudnn8-devel-ubuntu18.04`
  
 ## 模型说明
 
* 使用了SwinTransformer、CSwinTransformer等作为图像部分的编码器；
* 使用ERNIE-3.0、GPT等基于TansformerEncoder/Decoder的中文预训练模型作为TrOCR文本部分的解码器以适用于中文OCR识别任务；
* 可以改用较小的字典（如`ocr_keys_v1.txt`，包含6623个常见字符），脚本在加载预训练Decoder模型时会重构`wordembedding`以适应新的字典；
* 集成了 NVIDIA FasterTransformer 用于预测加速，以解决当模型解码器的维度、束搜索空间、层数，较高、大、深时可能出现的推断效率问题；
* 增加了一个自定义的[Beam Search方法](https://github.com/janelu9/TrOCR/blob/d3d3d7be156157ff802980a636a48aa29e4fc403/image2text.py#L757)，通常情况下更快更准。
 
`image2text.py`中包含了用于训练的Image2Text模型和用于快速推断的FasterTransformer, 模型基于paddlepaddle开发. 这时你可以从[paddlenlp模型库](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html)中加载各种基于中文数据集的预训练模型. 当处理OCR任务时其基本等同于微软研究院基于Fairseq开源的[TrOCR](https://www.msra.cn/zh-cn/news/features/trocr).
 
 *注：FasterTransformer会在**FasterTransformer**第一次被调用时自动编译。*
 
 ## 模型训练调优、评估和保存
 1. 在`train.py`中配置好数据目录、训练集标签、测试集标签和预训练模型位置等参数。
 2. 训练模型：
```
# 单机多卡启动，默认使用当前可见的所有卡
$ python -m paddle.distributed.launch train.py

# 单机多卡启动，设置当前使用的第0号和第1号卡
$ python -m paddle.distributed.launch --gpus '0,1' train.py

# 单机多卡启动，设置当前使用第0号和第1号卡
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m paddle.distributed.launch train.py
```	
*注：在部署推理服务时可以通过`paddle.jit.to_static`将模型转换为静态图（demo在`image2text.py`脚本的最后注释部分），然后使用[paddle inference](https://paddle-inference.readthedocs.io/en/latest/index.html)加载以提升推理效率。*
