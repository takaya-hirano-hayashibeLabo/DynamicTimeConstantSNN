from torch import nn
from snntorch import surrogate
import torch
from copy import deepcopy
from collections import OrderedDict

from .lif_model import DynamicLIF,LIF



def get_conv_outsize(model,in_size,in_channel):
    input_tensor = torch.randn(1, in_channel, in_size, in_size)
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape




class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3,stride=1,padding=1,num_block=1,bias=True,dropout=0.3):
        """
        CNNのみの残差ブロック
        このResidualでは入出力のサイズは変わらない (channelはもちろん変わる)
        :param num_block: 入力のCNN以外に何個のCNNを積むか
        """
        super(ResidualBlock,self).__init__()

        modules=[]
        modules+=[
                nn.Conv2d(
                    in_channels=in_channel,out_channels=out_channel,
                    kernel_size=kernel,stride=stride,padding=padding,bias=bias
                ),
                nn.ReLU(inplace=False)
            ]
        for _ in range(num_block):
            if dropout>0:
                modules+=[
                    nn.Dropout(dropout,inplace=False)
                ]
            modules+=[
                    nn.Conv2d(
                        in_channels=out_channel,out_channels=out_channel,
                        kernel_size=kernel,stride=stride,padding=padding,bias=bias
                    ),
                    nn.ReLU(inplace=False)
                ]
        self.model=nn.Sequential(*modules)

        self.shortcut=nn.Sequential()
        if not in_channel==out_channel:
            self.shortcut=nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,out_channels=out_channel,
                        kernel_size=1,stride=1,padding=0,bias=bias
                    )
            )

    def forward(self,x):
        """
        残差を足して出力
        """
        out=self.model(x)
        residual=self.shortcut(x)
        return out+residual



class ResidualLIFBlock(ResidualBlock):
    """
    活性化をLIFとした残差ブロック
    """

    def __init__(
            self,in_size:tuple,in_channel,out_channel,kernel=3,stride=1,padding=1,num_block=1,bias=False,dropout=0.3,
            dt=0.01,init_tau=0.5,min_tau=0.1,threshold=1.0,vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False,is_train_tau=True):
        """
        出力は電流(=スパイクではない)
        """
        super(ResidualLIFBlock,self).__init__(
            in_channel,out_channel,kernel,stride,padding,num_block,bias,dropout,
        )

        modules=[]
        modules+=[
                nn.Conv2d(
                    in_channels=in_channel,out_channels=out_channel,
                    kernel_size=kernel,stride=stride,padding=padding,bias=bias
                ),
            ]
        
        for _ in range(num_block):

            #blockの出力サイズを計算
            module_outsize=get_conv_outsize(nn.Sequential(*modules),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]
            modules.append(
                LIF(
                    in_size=tuple(module_outsize[1:]), dt=dt,
                    init_tau=init_tau, min_tau=min_tau,
                    threshold=threshold, vrest=vrest,
                    reset_mechanism=reset_mechanism, spike_grad=spike_grad,
                    output=False,is_train_tau=is_train_tau
                )
            )
            
            if dropout>0:
                modules+=[
                    nn.Dropout(dropout,inplace=False)
                ]
            modules+=[
                    nn.Conv2d(
                        in_channels=out_channel,out_channels=out_channel,
                        kernel_size=kernel,stride=stride,padding=padding,bias=bias
                    )
                ]
            
            
        self.model=nn.Sequential(*modules)

        self.shortcut=nn.Sequential()
        if not in_channel==out_channel:
            self.shortcut=nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,out_channels=out_channel,
                        kernel_size=1,stride=1,padding=0,bias=bias
                    )
            )


    def init_voltage(self):
        for layer in self.model:
            if isinstance(layer,LIF):
                layer.init_voltage()





class ResidualDynaLIFBlock(ResidualBlock):

    def __init__(
            self,in_size:tuple,in_channel,out_channel,kernel=3,stride=1,padding=1,num_block=1,bias=False,dropout=0.3,
            dt=0.01,init_tau=0.5,min_tau=0.1,threshold=1.0,vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False,
            v_actf=None
            ):
        """
        出力は電流(=スパイクではない)
        """
        super(ResidualDynaLIFBlock,self).__init__(
            in_channel,out_channel,kernel,stride,padding,num_block,bias,dropout,
        )

        modules=[]
        modules+=[
                nn.Conv2d(
                    in_channels=in_channel,out_channels=out_channel,
                    kernel_size=kernel,stride=stride,padding=padding,bias=bias
                ),
            ]
        
        for _ in range(num_block):

            #blockの出力サイズを計算
            module_outsize=get_conv_outsize(nn.Sequential(*modules),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]
            modules+=[
                DynamicLIF(
                    in_size=tuple(module_outsize[1:]), dt=dt,
                    init_tau=init_tau, min_tau=min_tau,
                    threshold=threshold, vrest=vrest,
                    reset_mechanism=reset_mechanism, spike_grad=spike_grad,
                    output=False, v_actf=v_actf
                )
            ]
            
            if dropout>0:
                modules+=[
                    nn.Dropout(dropout,inplace=False)
                ]
            modules+=[
                    nn.Conv2d(
                        in_channels=out_channel,out_channels=out_channel,
                        kernel_size=kernel,stride=stride,padding=padding,bias=bias
                    )
                ]
            
            
        self.model=nn.Sequential(*modules)

        self.shortcut=nn.Sequential()
        if not in_channel==out_channel:
            self.shortcut=nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,out_channels=out_channel,
                        kernel_size=1,stride=1,padding=0,bias=bias
                    )
            )


    def init_voltage(self):
        for layer in self.model:
            if isinstance(layer,DynamicLIF):
                layer.init_voltage()


    def set_dynamic_params(self,a):
        """
        LIFの時定数&膜抵抗を変動させる関数
        :param a: [スカラ]その瞬間の時間スケール
        """
        for layer in self.model:
            if isinstance(layer,DynamicLIF): #ラプラス変換によると時間スケールをかけると上手く行くはず
                layer.a = a


    def reset_params(self):
        for layer in self.model:
            if isinstance(layer,DynamicLIF):
                layer.a=1.0


    def get_tau(self):
        """
        tauを取得するだけ
        """

        taus={}
        layer_num=0
        for layer in self.model:
            if isinstance(layer,DynamicLIF):
                with torch.no_grad():
                    tau=layer.min_tau + layer.w.sigmoid()
                    # print(f"layer: Residual{layer._get_name()}-layer{layer_num}, tau shape: {tau.shape}")
                    taus["ResidualDynamicLIF"]=tau
                    layer_num+=1

        return taus


    def test_set_dynamic_param_list(self,a:list):
        """
        LIFの時定数&膜抵抗を変動させる関数
        :param a: [スカラ]その瞬間の時間スケール
        """
        a_list_tmp=deepcopy(a)
        for idx,layer in enumerate(self.model):
            if isinstance(layer,DynamicLIF): #ラプラス変換によると時間スケールをかけると上手く行くはず
                layer.a = a_list_tmp.pop(0)
        
        return a_list_tmp



