"""
時系列分散表現から連続値を予測するためのクラス
ここをattentionにしてもいいと思う
"""

import torch
import torch.nn as nn

from .dynamic_snn import DynamicSNN


class ContinuousSNN(nn.Module):

    def __init__(self,conf:dict, time_encoder:DynamicSNN):
        """
        :param conf: 連続値予測の出力層のconfig
        :param time_encoder: 時系列表現に落とすモデル(DynamicSNNとか)
        """
        super(ContinuousSNN,self).__init__()

        self.time_encoder=time_encoder

        self.in_size=conf["in-size"]
        self.out_size=conf["out-size"]
        self.hiddens=conf["hiddens"]
        self.dropout=conf["dropout"]
        self.clip_norm=conf["clip-norm"]
        if not "out-actf" in conf:
            self.out_actf=nn.Identity()
        elif conf["out-actf"]=="identity":
            self.out_actf=nn.Identity()
        elif conf["out-actf"]=="tanh":
            self.out_actf=nn.Tanh()
        else:
            raise ValueError(f"Invalid out-actf: {conf['out-actf']}")


        modules=[]
        modules+=[
            nn.Conv1d(in_channels=self.in_size, out_channels=self.hiddens[0],kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ]
        
        prev_hidden=self.hiddens[0]
        for i in range(len(self.hiddens)-1):
            hidden=self.hiddens[i+1]
            modules+=[
                nn.Conv1d(in_channels=prev_hidden,out_channels=hidden,kernel_size=1),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ]

        modules+=[
            nn.Conv1d(in_channels=self.hiddens[-1],out_channels=self.out_size,kernel_size=1),
            self.out_actf
        ]

        self.model=nn.Sequential(*modules)


    def clip_gradients(self):
        """
        Clips gradients to prevent exploding gradients.
        
        :param max_norm: Maximum norm of the gradients.
        """
        # Clip gradients for the ContiuousModel parameters
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        

    def forward(self,inspikes:torch.Tensor,return_v:bool=False):
        """
        :param inspikes: [N x T x xdim]
        """

        in_sp=inspikes.permute((1,0,*[i+2 for i in range(inspikes.ndim-2)])) #[T x N x xdim]
        _,_,out_v=self.time_encoder.forward(in_sp) #[T x N x outdim]

        out_v=out_v.permute(1,2,0) #[N x outdim x T] 時間方向は関連させない
        out:torch.Tensor=self.model(out_v) #[N x outdim x T]

        if return_v:
            return out.permute(0,2,1),out_v.permute(0,2,1)
        else:
            return out.permute(0,2,1) #[N x T x outdim]


    def dynamic_forward_given_scale(self,inspikes:torch.Tensor,scales:torch.Tensor,return_v:bool=False):

        with torch.no_grad():
            in_sp=inspikes.permute((1,0,*[i+2 for i in range(inspikes.ndim-2)])) #[T x N x xdim]
            _,_,out_v=self.time_encoder.dynamic_forward_v1(in_sp,scales)

            out_v=out_v/(scales.unsqueeze(-1).unsqueeze(-1))
            out_v=out_v.permute(1,2,0)
            out:torch.Tensor=self.model(out_v)

        if return_v:    
            return out.permute(0,2,1),out_v.permute(0,2,1)
        else:
            return out.permute(0,2,1)