import torch
import torch.nn as nn
from snntorch import surrogate
from math import log

class LIF(nn.Module):
    def __init__(
            self,in_size:tuple,dt,init_tau=0.5,min_tau=0.1,threshold=1.0,vrest=0,
            reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False,
            is_train_tau=True,reset_v=True
        ):
        """
        :param in_size: currentの入力サイズ
        :param dt: LIFモデルを差分方程式にしたときの⊿t. 元の入力がスパイク時系列ならもとデータと同じ⊿t. 
        :param init_tau: 膜電位時定数τの初期値
        :param threshold: 発火しきい値
        :param vrest: 静止膜電位. 
        :paarm reset_mechanism: 発火後の膜電位のリセット方法の指定
        :param spike_grad: 発火勾配の近似関数
        :param reset_v: 膜電位vをリセットするか (最終層のvを使うときだけFalseにしても良い)
        """
        super(LIF, self).__init__()

        self.dt=dt
        self.init_tau=init_tau
        self.min_tau=min_tau
        self.threshold=threshold
        self.vrest=vrest
        self.reset_mechanism=reset_mechanism
        self.spike_grad=spike_grad
        self.output=output
        self.v_peak=self.threshold*3
        self.reset_v=reset_v

        #>> tauを学習可能にするための調整 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 参考 [https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py]
        self.is_train_tau=is_train_tau
        init_w=-log(1/(self.init_tau-min_tau)-1)
        if is_train_tau:
            self.w=nn.Parameter(init_w * torch.ones(size=in_size))  # デフォルトの初期化
        elif not is_train_tau:
            self.w=(init_w * torch.ones(size=in_size))  # デフォルトの初期化
        #<< tauを学習可能にするための調整 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.v=0.0
        self.r=1.0 #膜抵抗


    def forward(self,current:torch.Tensor):
        """
        :param current: シナプス電流 [batch x ...]
        """


        # if torch.max(self.tau)<self.dt: #dt/tauが1を超えないようにする
        #     dtau=(self.tau<self.dt)*self.dt
        #     self.tau=self.tau-dtau

        # print(f"tau:{self.tau.shape}, v:{self.v.shape}, current:{current.shape}")
        # print(self.tau)
        # print(self.v)
        # print("--------------")
        device=current.device

        if not self.is_train_tau:
            self.w=self.w.to(device)

        # Move v and w to the same device as current if they are not already
        if isinstance(self.v,torch.Tensor):
            if self.v.device != device:
                self.v = self.v.to(device)
        if self.w.device != device:
            self.w = self.w.to(device)


        tau=self.min_tau+self.w.sigmoid() #tauが小さくなりすぎるとdt/tauが1を超えてしまう
        dv=self.dt/(tau) * ( -(self.v-self.vrest) + (self.r)*current ) #膜電位vの増分
        self.v=self.v+dv
        spike=self.__fire()

        if self.reset_v:
            v_tmp=self.v*(1.0-spike) + self.v_peak*spike #リセット前の膜電位
            self.__reset_voltage(spike)
        elif not self.reset_v:
            v_tmp=self.v #リセットしないときはピーク値への書き換えもしない

        if not self.output:
            return spike
        else:
            return spike, v_tmp


    def __fire(self):
        v_shift=self.v-self.threshold
        spike=self.spike_grad(v_shift)
        return spike
    

    def __reset_voltage(self,spike):
        if self.reset_mechanism=="zero":
            self.v=self.v*(1-spike.float())
        elif self.reset_mechanism=="subtract":
            self.v=self.v-self.threshold


    def init_voltage(self):
        if not self.v is None:
            self.v=0.0



class DynamicLIF(nn.Module):
    """
    動的にtime constant(TC)が変動するLIF
    """

    def __init__(
            self,in_size:tuple,dt,init_tau=0.5,min_tau=0.1,threshold=1.0,
            vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False, 
            reset_v=True, v_actf=None
        ):
        """
        :param in_size: currentの入力サイズ
        :param dt: LIFモデルを差分方程式にしたときの⊿t. 元の入力がスパイク時系列ならもとデータと同じ⊿t. 
        :param init_tau: 膜電位時定数τの初期値. [[tau1, rate1],[tau2, rate2],...]で指定することでその割合の指定が可能
        :param threshold: 発火しきい値
        :param vrest: 静止膜電位. 
        :paarm reset_mechanism: 発火後の膜電位のリセット方法の指定
        :param spike_grad: 発火勾配の近似関数
        :param reset_v: 膜電位vをリセットするか (最終層のvを使うときだけFalseにしても良い)
        :param v_actf<str>: {None, "relu", "tanh"} 膜電位vの活性化関数 (いい感じに内部状態を調整するために使ってる)
        """
        super(DynamicLIF, self).__init__()

        self.dt=dt
        self.init_tau=init_tau
        self.min_tau=min_tau
        self.threshold=threshold
        self.vrest=vrest
        self.reset_mechanism=reset_mechanism
        self.spike_grad=spike_grad
        self.output=output
        self.reset_v=reset_v
        self.v_peak=self.threshold*3.0 #ピーク値はしきい値の3倍位にしとく(正直, 描画用なのでどうでもいい)

        #>> tauを学習可能にするための調整 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 参考 [https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py]
        init_w=-log(1/(self.init_tau-min_tau)-1)
        self.w=nn.Parameter(init_w * torch.ones(size=in_size))  # デフォルトの初期化
        #<< tauを学習可能にするための調整 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        self.v=0.0
        self.r=1.0 #膜抵抗
        self.a=1.0 #タイムスケール

        self.__set_v_actf(v_actf) #膜電位vの活性化関数を設定

    def __set_v_actf(self,v_actf):
        if v_actf=="relu":
            self.v_actf=torch.nn.ReLU()
        elif v_actf=="tanh":
            self.v_actf=torch.nn.Tanh()
        else:
            self.v_actf=None


    def forward(self,current:torch.Tensor):
        """
        :param current: シナプス電流 [batch x ...]
        """


        device = current.device  # Get the device of the input tensor

        # Move v and w to the same device as current if they are not already
        if isinstance(self.v,torch.Tensor):
            if self.v.device != device:
                self.v = self.v.to(device)
        if self.w.device != device:
            self.w = self.w.to(device)
        # if self.wh.weight.device != device:
        #     self.wh=self.wh.to(device)

        tau=self.min_tau+self.w.sigmoid() #tauが小さくなりすぎるとdt/tauが1を超えてしまう

        tau_new=tau*self.a
        tau_new[tau_new<self.min_tau]=self.min_tau #tauが小さくなりすぎるとdt/tauが1を超えてしまうためclippingする
        
        dv=(self.dt/tau_new) * (-(self.v-self.vrest)) + (self.dt/(tau*self.a)) * (self.a*self.r*current) #内部状態に対するtauのみclipping
       
        self.v=self.v+dv

        if not self.v_actf is None: #活性化関数が設定されているなら適用
            self.v=self.v_actf(self.v)

        spike=self.__fire() 


        if self.reset_v: 
            v_tmp=self.v*(1.0-spike) + self.v_peak*spike #spikeがたった膜電位をピーク値にする(描画用)
            self.__reset_voltage(spike)
        else:
            v_tmp=self.v #リセットしないときはピーク値への書き換えもしない
        
        self._set_state(current,v_tmp,spike) #状態を記憶 見たいときにオンにする

        if not self.output:
            return spike
        else:
            return spike, (current,v_tmp) #spikeに加えて, (シナプス電流, 膜電位)を返す


    def __fire(self)->torch.Tensor:
        v_shift=self.v-self.threshold
        spike=self.spike_grad(v_shift)
        return spike
    

    def __reset_voltage(self,spike):
        
        if self.reset_mechanism=="zero":
            self.v=self.v*(1-spike.float())
        elif self.reset_mechanism=="subtract":
            self.v=self.v-self.threshold


    def init_voltage(self):
        self.spike_prev=None #spikeのリセット. ホントは他の関数にしたほうがいい(読みやすいコード的に)
        self.v=0.0

    def _set_state(self,current:torch.Tensor,volt:torch.Tensor,outspike:torch.Tensor):
        """
        各時刻の各層のデータを保存したいときに使う
        """
        self.lif_state={
            "current":current,
            "volt":volt,
            "outspike":outspike
        }



