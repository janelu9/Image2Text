# Image2TextModel
An image to text model base on transformer which can also be used on OCR task.

在这里我们集成了 NVIDIA [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1) 用于预测加速。同时集成了 FasterTransformer float32 以及 float16 预测。以下是使用 FasterTransformer 的说明。

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
  
 ## Model说明
 脚本`Image2Text.py`内包含了用于训练的Image2Text模型和用于快速推断的FastDecoder. 当处理OCR任务时其等同于微软研究院基于Fairseq开源的[TrOCR](https://www.msra.cn/zh-cn/news/features/trocr). 
 
 *FasterTransformer会在FastDecoder第一次被调用时自动编译.*
