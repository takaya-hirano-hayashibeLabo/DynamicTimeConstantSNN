import torch
import torch.nn as nn
from snntorch import surrogate
from math import log

from .residual_block import ResidualBlock, ResidualLIFBlock
from .lif_model import LIF


class SNN(nn.Module):
    """
    SNN with dynamic time constant(TC)
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
        self.reset_outv=conf["reset-outmem"] if "reset-outmem" in conf.keys() else True #whether to reset the last layer's LIF


        modules=[]
        is_bias=True

        #>> input layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Linear(self.in_size, self.hiddens[0],bias=is_bias),
            LIF(
                in_size=(self.hiddens[0],),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False,is_train_tau=self.is_train_tau
            ),
            nn.Dropout(self.dropout),
        ]
        #<< input layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> intermediate layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
        #<< intermediate layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> output layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Linear(self.hiddens[-1], self.out_size,bias=is_bias),

            # give the reset bool value of the membrane potential to the last layer's LIF
            LIF(
                in_size=(self.out_size,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True,is_train_tau=self.is_train_tau,reset_v=self.reset_outv
            ),
        ]
        #<< output layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.model=nn.Sequential(*modules)


    def __init_lif(self):
        for layer in self.model:
            if isinstance(layer,LIF) or isinstance(layer,ResidualLIFBlock):
                layer.init_voltage()


    def forward(self,s:torch.Tensor):
        """
        :param s: spike sequence [T x batch x ...]
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
        split the parameters to be applied and not applied weight decay
        L2 regularization is basically applied to weights only (not to bias or time constants)
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
    param: in_size: width and height (square)
    param: in_channel: channel size
    param: out_channel: output channel size
    param: kernel: kernel size
    param: stride: stride size
    param: padding: padding size
    param: is_bias: whether to use bias
    param: is_bn: whether to use batch normalization
    param: pool_type: pooling type ("avg" or "max")
    param: dropout: dropout rate
    param: lif_dt: LIF model time step
    param: lif_init_tau: LIF initial time constant
    param: lif_min_tau: LIF minimum time constant
    param: lif_threshold: LIF firing threshold
    param: lif_vrest: LIF resting membrane potential
    param: lif_reset_mechanism: LIF membrane potential reset mechanism
    param: lif_spike_grad: LIF spike gradient function
    param: lif_output: whether to return the output of LIF
    param: is_train_tau: whether to learn the time constant of LIF
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

    #calculate the output size of the block
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

        #>> convolution layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        in_c=self.in_channel
        in_size=self.in_size
        for i,hidden_c in enumerate(self.hiddens):

            block,block_outsize=add_csnn_block(
                in_size=in_size, in_channel=in_c, out_channel=hidden_c,
                kernel=3, stride=1, padding=1,  # kernel size, stride, padding
                is_bias=True, is_bn=self.is_bn, pool_type=self.pool_type, pool_size=self.pool_size[i] , dropout=self.dropout,
                lif_dt=self.dt, lif_init_tau=self.init_tau, lif_min_tau=self.min_tau,
                lif_threshold=self.v_threshold, lif_vrest=self.v_rest,
                lif_reset_mechanism=self.reset_mechanism, lif_spike_grad=self.spike_grad,
                lif_output=False,  # do not return the output
                is_train_tau=self.is_train_tau
            )
            modules+=block
            in_c=hidden_c
            in_size=block_outsize[-1]
        #<< convolution layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> linear layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
        #<< linear layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        self.model=nn.Sequential(*modules)


def add_residual_block(
        in_size,in_channel,out_channel,kernel,stride,padding,is_bias,residual_block_num,is_bn,pool_type,pool_size,dropout,
        lif_dt,lif_init_tau,lif_min_tau,lif_threshold,lif_vrest,lif_reset_mechanism,lif_spike_grad,lif_output, is_train_tau,
        res_actfn
        ):
    """
    param: in_size: width and height (square)
    param: in_channel: channel size
    param: out_channel: output channel size
    param: kernel: kernel size
    param: stride: stride size
    param: padding: padding size
    param: is_bias: whether to use bias
    param: residual_block_num: number of CNN in ResBlock (0 is also OK)
    param: is_bn: whether to use batch normalization
    param: pool_type: pooling type ("avg" or "max")
    param: pool_size: pool size
    param: dropout: dropout rate
    param: lif_dt: LIF model time step
    param: lif_init_tau: LIF initial time constant
    param: lif_min_tau: LIF minimum time constant
    param: lif_threshold: LIF firing threshold
    param: lif_vrest: LIF resting membrane potential
    param: lif_reset_mechanism: LIF membrane potential reset mechanism
    param: lif_spike_grad: LIF spike gradient function
    param: lif_output: whether to return the output of LIF
    param: is_train_tau: whether to learn the time constant of LIF
    param: res_actfn: residual block activation function {relu, lif}
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

    #calculate the output size of the block
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
    SNN with residual CNN
    """
    def __init__(self,conf):
        super(ResCSNN,self).__init__(conf)

        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.out_size = conf["out-size"]
        self.hiddens = conf["hiddens"]
        self.residual_blocks=conf["residual-block"] #number of CNN in each residual block
        self.pool_type = conf["pool-type"]
        self.pool_size=conf["pool-size"]
        self.is_bn = conf["is-bn"]
        self.linear_hidden = conf["linear-hidden"]
        self.dropout = conf["dropout"]
        self.res_actfn=conf["res-actfn"] if "res-actfn" in conf.keys() else "relu" #residual block activation function

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

        is_bias=conf["is-bias"] if "is-bias" in conf.keys() else True #whether to use bias for CNN

        modules=[]

        #>> convolution layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        in_c=self.in_channel
        in_size=self.in_size
        for i,hidden_c in enumerate(self.hiddens):

            block,block_outsize=add_residual_block(
                res_actfn=self.res_actfn,
                in_size=in_size, in_channel=in_c, out_channel=hidden_c,
                kernel=3, stride=1, padding=1,  # kernel size, stride, padding
                is_bias=is_bias, residual_block_num=self.residual_blocks[i],
                is_bn=self.is_bn, pool_type=self.pool_type,pool_size=self.pool_size[i],dropout=self.dropout,
                lif_dt=self.dt, lif_init_tau=self.init_tau, lif_min_tau=self.min_tau,
                lif_threshold=self.v_threshold, lif_vrest=self.v_rest,
                lif_reset_mechanism=self.reset_mechanism, lif_spike_grad=self.spike_grad,
                lif_output=False, is_train_tau=self.is_train_tau  # do not return the output
            )
            modules+=block
            in_c=hidden_c
            in_size=block_outsize[-1]
        #<< convolution layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> linear layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
        #<< linear layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        self.model=nn.Sequential(*modules)
