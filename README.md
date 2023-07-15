# Image2Text
An image to text model based on transformer used on OCR task.

在这里集成了 NVIDIA [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1) 用于预测加速。同时集成了 FasterTransformer float32 以及 float16 预测。以下是使用 FasterTransformer 的说明。

## 环境说明

* 本项目依赖于 PaddlePaddle 2.3.2 GPU版本
* 如果使用NVIDIA FasterTransformer则需安装如下依赖：

	CMake >= 3.10
	
	CUDA 10.1 或以上版本 （需要 PaddlePaddle 框架一致）
	
	gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
	
	推荐使用 Python3，python依赖：
	
	 ```
	 pip install attrdict pyyaml paddlenlp
	 ```
* 对于图灵架构GPU(CUDA版本为10.1)也可以通过docker镜像来安装paddle运行环境：

拉取cuda运行环境镜像（已经配置好对应的gcc和cmake等常用工具）
```shell
docker pull registry.cn-shanghai.aliyuncs.com/janelu9/dl:cuda-10.1-cudnn8-ubuntu18.04
```
启动容器
```shell
docker run -d --gpus all --name paddle --shm-size=32g --ulimit memlock=-1 -v /mnt:/mnt -p 8888:8888 -it 146273117745 /bin/bash
```
安装[Anaconda](https://repo.anaconda.com/archive/)
```shell
./Anaconda3-*-Linux-x86_64.sh
```
```shell
source ~/.bashrc
```
安装paddle及其依赖
```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
```shell
pip install paddlepaddle-gpu==2.3.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
 ## 模型说明

* 使用了SwinTransformer、CSwinTransformer等作为图像部分的编码器；

* 使用了KLDivLoss和CTCLoss相结合的损失函数（可以仅使用Attention解码，也可以使用CTC前缀束搜索加Attention重排序的方式解码）；

* 使用ERNIE-3.0等基于TansformerEncoder/Decoder的中文预训练模型作为TrOCR文本部分的解码器以适用于中文OCR识别任务；

* 可以改用较小的字典（如`ocr_keys_v1.txt`，包含6623个常见字符），脚本在加载预训练Decoder模型时会重构`wordembedding`以适应新的字典；

* 集成了 NVIDIA FasterTransformer 用于预测加速，以解决当模型解码器的维度、束搜索空间、层数，较高、大、深时可能出现的推断效率问题；

* 增加了一个自定义的[Beam Search方法](https://github.com/janelu9/TrOCR/blob/main/image2text.py#L760)，通常情况下更快更准。

`image2text.py`中包含了用于训练的Image2Text模型和用于快速推断的FasterTransformer, 模型基于paddlepaddle开发. 这时你可以从[paddlenlp模型库](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/#id2)中加载各种基于中文数据集的预训练模型.

`image_aug.py`中包含了多种数据增强策略，可以有效增加模型泛化能力。用户也可以自行添加基于pillow或opencv的图像增强策略，注意好img和ndarray间的转换即可。

 *注：FasterTransformer会在**FasterTransformer**第一次被调用时自动编译。*

 ## 模型训练调优、评估、保存和推理部署
 1. 在`train.py`中配置好数据目录、训练集标签、测试集标签和预训练模型位置等参数。
 2. 训练模型、评估和保存：

单机多卡启动，默认使用当前可见的所有卡
```shell
python -m paddle.distributed.launch train.py
```
单机多卡启动，设置当前使用的第0号和第1号卡
```shell
python -m paddle.distributed.launch --gpus '0,1' train.py
```
单机多卡启动，设置当前使用第0号和第1号卡
```shell
export CUDA_VISIBLE_DEVICES=0,1 && python -m paddle.distributed.launch train.py
```
模型训练过程会自动保存验证集效果最好的参数于文件`best_model.pdparams`中。

 3. 在部署推理服务时可以通过`paddle.jit`将模型转换为静态图，然后使用[paddle inference](https://paddle-inference.readthedocs.io/en/latest/index.html)加载以提升推理效率。

```shell
paddle.jit.save(layer=infer,path="inference",input_spec=[paddle.static.InputSpec(shape=[None,3,384,384],dtype='float32')])
```

*注：该模型在用于处理不规则排列文字的OCR任务（如印章、公式、多行文本识别）时简单有效。准确率在样本呈正态分布的测试集上（比较客观的抽样样本集上）通常可以达到90%以上。*

## Citation

If you find it useful or use my TrOCR code  in your research, please cite it in your publications.

```bibtex
@misc{EasyLLM,
  author       = {Jian Lu},
  title        = {TrOCR: An image to text model based on transformer used on OCR task},
  year         = {2022},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://gitee.com/janelu9/TrOCR.git}},
}
```

