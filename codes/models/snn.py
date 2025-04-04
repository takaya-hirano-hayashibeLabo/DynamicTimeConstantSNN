import torch
import torch.nn as nn
from snntorch import surrogate
from math import log

from .residual_block import ResidualBlock, ResidualLIFBlock
from .lif_model import LIF


class SNN(nn.Module):
    """
    time constant(TC)が動的に変動するSNN
    """

    def __init__(self,conf:dict):
        super(SNN,self).__init__()

        self.in_size=conf["in-size"]
        self.hiddens = conf["hiddens"]
        self.out_size = conf["out-size"]
        self.clip_norm=conf["clip-norm"] if "clip-norm" in conf.keys() else 1.0
        self.dropout=conf["dropout"]
        self.output_mem=conf["output-membrane"]

        self.dt = conf["dt"]
        self.init_tau = conf["init-tau"]
        self.min_tau=conf["min-tau"]
        self.is_train_tau=conf["train-tau"]
        self.v_threshold = conf["v-threshold"]
        self.v_rest = conf["v-rest"]
        self.reset_mechanism = conf["reset-mechanism"]
        if "fast".casefold() in conf["spike-grad"] and "sigmoid".casefold() in conf["spike-grad"]: 
            self.spike_grad = surrogate.fast_sigmoid()
        self.reset_outv=conf["reset-outmem"] if "reset-outmem" in conf.keys() else True #最終層のLifをリセットするか


        modules=[]
        is_bias=True

        #>> 入力層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Linear(self.in_size, self.hiddens[0],bias=is_bias),
            LIF(
                in_size=(self.hiddens[0],),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False,is_train_tau=self.is_train_tau
            ),
            nn.Dropout(self.dropout),
        ]
        #<< 入力層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> 中間層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        prev_hidden=self.hiddens[0]
        for hidden in self.hiddens[1:]:
            modules+=[
                nn.Linear(prev_hidden, hidden,bias=is_bias),
                LIF(
                    in_size=(hidden,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                    reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False,is_train_tau=self.is_train_tau
                ),
                nn.Dropout(self.dropout),
            ]
            prev_hidden=hidden
        #<< 中間層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> 出力層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Linear(self.hiddens[-1], self.out_size,bias=is_bias),

            # 出力層のLIFだけ, 膜電位のリセットbool値を与える
            LIF(
                in_size=(self.out_size,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True,is_train_tau=self.is_train_tau,reset_v=self.reset_outv
            ),
        ]
        #<< 出力層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.model=nn.Sequential(*modules)


    def __init_lif(self):
        for layer in self.model:
            if isinstance(layer,LIF) or isinstance(layer,ResidualLIFBlock):
                layer.init_voltage()


    def forward(self,s:torch.Tensor):
        """
        :param s: スパイク列 [T x batch x ...]
        :return out_s: [T x batch x ...]
        :return out_v: [T x batch x ...]
        """

        T=s.shape[0]
        self.__init_lif()

        out_s,out_v=[],[]
        for st in s:
            st_out,vt_out=self.model(st)
            out_s.append(st_out)
            out_v.append(vt_out)

        out_s=torch.stack(out_s,dim=0)
        out_v=torch.stack(out_v,dim=0)

        if self.output_mem:
            return out_s,[],out_v
        
        elif not self.output_mem:
            return out_s

    def clip_gradients(self):
        """
        Clips gradients to prevent exploding gradients.
        
        :param max_norm: Maximum norm of the gradients.
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)


    def split_weight_decay_params(self, no_decay_param_names:list=["w","bias"], weight_decay:float=0.01):
        """
        weight decayを適用するパラメータと適用しないパラメータを分ける  
        L2正則化は基本的に重みのみ (biasや時定数には適用しない)
        """
        decay_params=[]
        no_decay_params=[]
        for name,param in self.model.named_parameters():
            if any(nd in name for nd in no_decay_param_names):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        split_params=[
            {"params":decay_params,"weight_decay":weight_decay},
            {"params":no_decay_params,"weight_decay":0.0}
        ]

        return split_params


def get_conv_outsize(model,in_size,in_channel):
    input_tensor = torch.randn(1, in_channel, in_size, in_size)
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape



def add_csnn_block(
        in_size,in_channel,out_channel,kernel,stride,padding,is_bias,is_bn,pool_type,pool_size,dropout,
        lif_dt,lif_init_tau,lif_min_tau,lif_threshold,lif_vrest,lif_reset_mechanism,lif_spike_grad,lif_output,is_train_tau
        ):
    """
    param: in_size: 幅と高さ (正方形とする)
    param: in_channel: channel size
    param: out_channel: 出力チャネルのサイズ
    param: kernel: カーネルサイズ
    param: stride: ストライドのサイズ
    param: padding: パディングのサイズ
    param: is_bias: バイアスを使用するかどうか
    param: is_bn: バッチ正規化を使用するかどうか
    param: pool_type: プーリングの種類 ("avg"または"max")
    param: dropout: dropout rate
    param: lif_dt: LIFモデルの時間刻み
    param: lif_init_tau: LIFの初期時定数
    param: lif_min_tau: LIFの最小時定数
    param: lif_threshold: LIFの発火しきい値
    param: lif_vrest: LIFの静止膜電位
    param: lif_reset_mechanism: LIFの膜電位リセットメカニズム
    param: lif_spike_grad: LIFのスパイク勾配関数
    param: lif_output: LIFの出力を返すかどうか
    param: is_train_tau: LIFのtrain_tauを学習させるかどうか
    """
    
    block=[]
    block.append(
        nn.Conv2d(
            in_channels=in_channel,out_channels=out_channel,
            kernel_size=kernel,stride=stride,padding=padding,bias=is_bias
        )
    )

    if is_bn:
        block.append(
            nn.BatchNorm2d(out_channel)
        )

    if pool_type=="avg".casefold():
        block.append(nn.AvgPool2d(pool_size))
    elif pool_type=="max".casefold():
        block.append(nn.MaxPool2d(pool_size))

    #blockの出力サイズを計算
    block_outsize=get_conv_outsize(nn.Sequential(*block),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]

    block.append(
        LIF(
            in_size=tuple(block_outsize[1:]), dt=lif_dt,
            init_tau=lif_init_tau, min_tau=lif_min_tau,
            threshold=lif_threshold, vrest=lif_vrest,
            reset_mechanism=lif_reset_mechanism, spike_grad=lif_spike_grad,
            output=lif_output,is_train_tau=is_train_tau
        )
    )

    if dropout>0:
        block.append(nn.Dropout2d(dropout))
    
    return block, block_outsize




class CSNN(SNN):
    def __init__(self,conf):
        super(CSNN,self).__init__(conf)

        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.out_size = conf["out-size"]
        self.hiddens = conf["hiddens"]
        self.pool_type = conf["pool-type"]
        self.pool_size=conf["pool-size"]
        self.is_bn = conf["is-bn"]
        self.linear_hidden = conf["linear-hidden"]
        self.dropout = conf["dropout"]
        
        self.output_mem=conf["output-membrane"]
        self.dt = conf["dt"]
        self.init_tau = conf["init-tau"]
        self.min_tau=conf["min-tau"]
        self.is_train_tau=conf["train-tau"]
        self.v_threshold = conf["v-threshold"]
        self.v_rest = conf["v-rest"]
        self.reset_mechanism = conf["reset-mechanism"]
        if "fast".casefold() in conf["spike-grad"] and "sigmoid".casefold() in conf["spike-grad"]: 
            self.spike_grad = surrogate.fast_sigmoid()


        modules=[]

        #>> 畳み込み層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        in_c=self.in_channel
        in_size=self.in_size
        for i,hidden_c in enumerate(self.hiddens):

            block,block_outsize=add_csnn_block(
                in_size=in_size, in_channel=in_c, out_channel=hidden_c,
                kernel=3, stride=1, padding=1,  # カーネルサイズ、ストライド、パディングの設定
                is_bias=True, is_bn=self.is_bn, pool_type=self.pool_type, pool_size=self.pool_size[i] , dropout=self.dropout,
                lif_dt=self.dt, lif_init_tau=self.init_tau, lif_min_tau=self.min_tau,
                lif_threshold=self.v_threshold, lif_vrest=self.v_rest,
                lif_reset_mechanism=self.reset_mechanism, lif_spike_grad=self.spike_grad,
                lif_output=False,  # 出力を返さない設定
                is_train_tau=self.is_train_tau
            )
            modules+=block
            in_c=hidden_c
            in_size=block_outsize[-1]
        #<< 畳み込み層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> 線形層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Flatten(),
            nn.Linear(block_outsize[1]*block_outsize[2]*block_outsize[3],self.linear_hidden,bias=True),
            LIF(
                in_size=(self.linear_hidden,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False,is_train_tau=self.is_train_tau
            ),
            nn.Linear(self.linear_hidden,self.out_size,bias=True),
            LIF(
                in_size=(self.out_size,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True,is_train_tau=self.is_train_tau
            ),
        ]
        #<< 線形層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        self.model=nn.Sequential(*modules)


def add_residual_block(
        in_size,in_channel,out_channel,kernel,stride,padding,is_bias,residual_block_num,is_bn,pool_type,pool_size,dropout,
        lif_dt,lif_init_tau,lif_min_tau,lif_threshold,lif_vrest,lif_reset_mechanism,lif_spike_grad,lif_output, is_train_tau,
        res_actfn
        ):
    """
    param: in_size: 幅と高さ (正方形とする)
    param: in_channel: channel size
    param: out_channel: 出力チャネルのサイズ
    param: kernel: カーネルサイズ
    param: stride: ストライドのサイズ
    param: padding: パディングのサイズ
    param: is_bias: バイアスを使用するかどうか
    param: residual_block_num: ResBlock内のCNNの数 (0でもいい)
    param: is_bn: バッチ正規化を使用するかどうか
    param: pool_type: プーリングの種類 ("avg"または"max")
    param: pool_size: プールのサイズ
    param: dropout: dropout rate
    param: lif_dt: LIFモデルの時間刻み
    param: lif_init_tau: LIFの初期時定数
    param: lif_min_tau: LIFの最小時定数
    param: lif_threshold: LIFの発火しきい値
    param: lif_vrest: LIFの静止膜電位
    param: lif_reset_mechanism: LIFの膜電位リセットメカニズム
    param: lif_spike_grad: LIFのスパイク勾配関数
    param: lif_output: LIFの出力を返すかどうか
    param: is_train_tau: LIFのtauを学習するかどうか
    param: res_actfn: 残差ブロックの活性化関数 {relu, lif}
    """
    
    block=[]
    if res_actfn=="relu".casefold():
        block.append(
            ResidualBlock(
                in_channel=in_channel,out_channel=out_channel,
                kernel=kernel,stride=stride,padding=padding,
                num_block=residual_block_num,bias=is_bias
            )
        )
    elif res_actfn=="lif".casefold():
        block.append(
            ResidualLIFBlock(
                in_channel=in_channel,out_channel=out_channel,
                kernel=kernel,stride=stride,padding=padding,
                num_block=residual_block_num,bias=is_bias,
                in_size=in_size, dt=lif_dt,
                init_tau=lif_init_tau, min_tau=lif_min_tau,
                threshold=lif_threshold, vrest=lif_vrest,
                reset_mechanism=lif_reset_mechanism, spike_grad=surrogate.fast_sigmoid(),
                output=False,is_train_tau=is_train_tau
            )
        )

    if is_bn:
        block.append(
            nn.BatchNorm2d(out_channel)
        )

    if pool_size>0:
        if pool_type=="avg".casefold():
            block.append(nn.AvgPool2d(pool_size))
        elif pool_type=="max".casefold():
            block.append(nn.MaxPool2d(pool_size))

    #blockの出力サイズを計算
    block_outsize=get_conv_outsize(nn.Sequential(*block),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]

    block.append(
        LIF(
            in_size=tuple(block_outsize[1:]), dt=lif_dt,
            init_tau=lif_init_tau, min_tau=lif_min_tau,
            threshold=lif_threshold, vrest=lif_vrest,
            reset_mechanism=lif_reset_mechanism, spike_grad=lif_spike_grad,
            output=lif_output, is_train_tau=is_train_tau
        )
    )

    if dropout>0:
        block.append(nn.Dropout2d(dropout, inplace=False))
    
    return block, block_outsize


class ResCSNN(SNN):
    """
    CNNを残差にしたSNN
    """
    def __init__(self,conf):
        super(ResCSNN,self).__init__(conf)

        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.out_size = conf["out-size"]
        self.hiddens = conf["hiddens"]
        self.residual_blocks=conf["residual-block"] #残差ブロックごとのCNNの数
        self.pool_type = conf["pool-type"]
        self.pool_size=conf["pool-size"]
        self.is_bn = conf["is-bn"]
        self.linear_hidden = conf["linear-hidden"]
        self.dropout = conf["dropout"]
        self.res_actfn=conf["res-actfn"] if "res-actfn" in conf.keys() else "relu" #残差ブロックの活性化関数

        self.output_mem=conf["output-membrane"]
        self.dt = conf["dt"]
        self.init_tau = conf["init-tau"]
        self.min_tau=conf["min-tau"]
        self.is_train_tau=conf["train-tau"]
        self.v_threshold = conf["v-threshold"]
        self.v_rest = conf["v-rest"]
        self.reset_mechanism = conf["reset-mechanism"]
        if "fast".casefold() in conf["spike-grad"] and "sigmoid".casefold() in conf["spike-grad"]: 
            self.spike_grad = surrogate.fast_sigmoid()

        is_bias=conf["is-bias"] if "is-bias" in conf.keys() else True #CNNバイアスをつけるかどうか

        modules=[]

        #>> 畳み込み層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        in_c=self.in_channel
        in_size=self.in_size
        for i,hidden_c in enumerate(self.hiddens):

            block,block_outsize=add_residual_block(
                res_actfn=self.res_actfn,
                in_size=in_size, in_channel=in_c, out_channel=hidden_c,
                kernel=3, stride=1, padding=1,  # カーネルサイズ、ストライド、パディングの設定
                is_bias=is_bias, residual_block_num=self.residual_blocks[i],
                is_bn=self.is_bn, pool_type=self.pool_type,pool_size=self.pool_size[i],dropout=self.dropout,
                lif_dt=self.dt, lif_init_tau=self.init_tau, lif_min_tau=self.min_tau,
                lif_threshold=self.v_threshold, lif_vrest=self.v_rest,
                lif_reset_mechanism=self.reset_mechanism, lif_spike_grad=self.spike_grad,
                lif_output=False, is_train_tau=self.is_train_tau  # 出力を返さない設定
            )
            modules+=block
            in_c=hidden_c
            in_size=block_outsize[-1]
        #<< 畳み込み層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> 線形層 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Flatten(),
            nn.Linear(block_outsize[1]*block_outsize[2]*block_outsize[3],self.linear_hidden,bias=is_bias),
            LIF(
                in_size=(self.linear_hidden,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False,is_train_tau=self.is_train_tau
            ),
            nn.Linear(self.linear_hidden,self.out_size,bias=is_bias),
            LIF(
                in_size=(self.out_size,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True,is_train_tau=self.is_train_tau
            ),
        ]
        #<< 線形層 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        self.model=nn.Sequential(*modules)
