import math
from functools import partial

import paddle
import numpy as np
from paddle import nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer

##Swish激活函数  由之前的激活函数复合而成出来的   
##通过创建 PyLayer 子类的方式实现Python端自定义算子
class SwishImplementation(PyLayer):
    def forward(ctx, i):
        result = i * F.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = F.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Layer):
    def forward(self, x):
        return SwishImplementation.apply(x)


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += paddle.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = paddle.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_width_and_height_from_size(x):
    """ Obtains width and height from a int or tuple """
    if isinstance(x, int): return x, x
    if isinstance(x, list) or isinstance(x, tuple): return x
    else: raise TypeError()

def calculate_output_image_size(input_image_size, stride):
    """
    计算出 Conv2dSamePadding with a stride.
    """
    if input_image_size is None: return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]



class Conv2DStaticSamePadding(nn.Conv2D):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2D((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2D(x, self.weight,self.stride, self.padding, self.dilation, self.groups)
        return x

class Identity(nn.Layer):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# MBConvBlock
class MBConvBlock(nn.Layer):
    '''
    层 ksize3*3 输入32 输出16  conv1  stride步长1
    '''
    def __init__(self, ksize, input_filters, output_filters, expand_ratio=1, stride=1, image_size=224):
        super().__init__()
        self._bn_mom = 0.1
        self._bn_eps = 0.01
        self._se_ratio = 0.25
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._kernel_size = ksize
        self._stride = stride

        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        if self._expand_ratio != 1:
            #Conv2D = get_same_padding_conv2D(image_size=image_size)
            self._expand_conv = Conv2D(in_channels=inp, out_channels=oup, kernel_size=1, padding='same')
            self._bn0 = nn.BatchNorm2D(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)


        # Depthwise convolution
        k = self._kernel_size
        s = self._stride
        #Conv2D = get_same_padding_conv2D(image_size=image_size)
        self._depthwise_conv = nn.Conv2D(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, padding='same')
        self._bn1 = nn.BatchNorm2D(num_features=oup, momentum=self._bn_mom)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        #Conv2D = get_same_padding_conv2D(image_size=(1,1))
        num_squeezed_channels = max(1, int(self._input_filters * self._se_ratio))
        self._se_reduce = nn.Conv2D(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1,padding='same')
        self._se_expand = nn.Conv2D(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1,padding='same')

        # Output phase
        final_oup = self._output_filters
        #Conv2D = get_same_padding_conv2D(image_size=image_size)
        self._project_conv = nn.Conv2D(in_channels=oup, out_channels=final_oup, kernel_size=1, padding='same')
        self._bn2 = nn.BatchNorm2D(num_features=final_oup, momentum=self._bn_mom)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._expand_ratio != 1:
            expand = self._expand_conv(inputs)
            bn0 = self._bn0(expand)
            x = self._swish(bn0)
        depthwise = self._depthwise_conv(x)
        bn1 = self._bn1(depthwise)
        x = self._swish(bn1)

        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = F.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._input_filters, self._output_filters
        if self._stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


# ScaledDotProductAttention结构



class ScaledDotProductAttention(nn.Layer):
    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).reshape([b_s, nq, self.h, self.d_k]).transpose([0, 2, 1, 3])  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).reshape([b_s, nk, self.h, self.d_k]).transpose([0, 2, 3, 1])  # (b_s, h, d_k, nk)
        v = self.fc_v(values).reshape([b_s, nk, self.h, self.d_v]).transpose([0, 2, 1, 3])  # (b_s, h, nk, d_v)

        att = q.matmul(k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = F.softmax(att, -1)
        att=self.dropout(att)
        out = (att.matmul(v)).transpose([0,2,1,3]).reshape([b_s, nq, self.h * self.d_v])  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



# CoAtNet网络结构

class CoAtNet(nn.Layer):
    def __init__(self,in_ch,image_size,out_chs=[64,96,192,384,768]):
        super().__init__()
        self.out_chs=out_chs
        self.maxpool2D = nn.MaxPool2D(kernel_size=2,stride=2)
        self.maxpool1D = nn.MaxPool1D(kernel_size=2, stride=2)

        self.s0=nn.Sequential(
            nn.Conv2D(in_ch,in_ch,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2D(in_ch,in_ch,kernel_size=3,padding=1)
        )
        self.mlp0=nn.Sequential(
            nn.Conv2D(in_ch,out_chs[0],kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(out_chs[0],out_chs[0],kernel_size=1)
        )
        
        self.s1=MBConvBlock(ksize=3,input_filters=out_chs[0],output_filters=out_chs[0],image_size=image_size//2)
        self.mlp1=nn.Sequential(
            nn.Conv2D(out_chs[0],out_chs[1],kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(out_chs[1],out_chs[1],kernel_size=1)
        )

        self.s2=MBConvBlock(ksize=3,input_filters=out_chs[1],output_filters=out_chs[1],image_size=image_size//4)
        self.mlp2=nn.Sequential(
            nn.Conv2D(out_chs[1],out_chs[2],kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(out_chs[2],out_chs[2],kernel_size=1)
        )

        self.s3=ScaledDotProductAttention(out_chs[2],out_chs[2]//8,out_chs[2]//8,8)
        self.mlp3=nn.Sequential(
            nn.Linear(out_chs[2],out_chs[3]),
            nn.ReLU(),
            nn.Linear(out_chs[3],out_chs[3])
        )

        self.s4=ScaledDotProductAttention(out_chs[3],out_chs[3]//8,out_chs[3]//8,8)
        self.mlp4=nn.Sequential(
            nn.Linear(out_chs[3],out_chs[4]),
            nn.ReLU(),
            nn.Linear(out_chs[4],out_chs[4])
        )


    def forward(self, x) :
        B,C,H,W=x.shape
        #stage0
        y=self.mlp0(self.s0(x))
        y=self.maxpool2D(y)
        #stage1
        y=self.mlp1(self.s1(y))
        y=self.maxpool2D(y)
        #stage2
        y=self.mlp2(self.s2(y))
        y=self.maxpool2D(y)
        #stage3
        y=y.reshape([B,self.out_chs[2],-1]).transpose([0,2,1]) #B,N,C
        y=self.mlp3(self.s3(y,y,y))
        y=self.maxpool1D(y.transpose([0,2,1])).transpose([0,2,1])
        #stage4
        y=self.mlp4(self.s4(y,y,y))
        y=self.maxpool1D(y.transpose([0,2,1])).transpose([0,2,1])
#         N=y.shape[-1]
#         y=y.reshape([B,self.out_chs[4],int(sqrt(N)),int(sqrt(N))])

        return y
