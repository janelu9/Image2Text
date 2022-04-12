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
  
 ## Model说明
 `image2text.py`中包含了用于训练的Image2Text模型和用于快速推断的FasterTransformer, 模型基于paddlepaddle开发. 这时你可以从[paddlenlp模型库](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html)中加载各种基于中文数据集的预训练模型. 
  当处理OCR任务时其等同于微软研究院基于Fairseq开源的[TrOCR](https://www.msra.cn/zh-cn/news/features/trocr). 结合[paddlepaddle的使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)，You can quickly use the Chinese version of TrOCR to fine tune your model now !
 
 *注：FasterTransformer会在**FasterTransformer**第一次被调用时自动编译.*
