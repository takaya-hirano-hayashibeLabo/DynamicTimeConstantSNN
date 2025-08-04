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
        :param in_size: input size of current
        :param dt: ⊿t when the LIF model is converted to a difference equation. If the input is a spike time series, it is the same as the original data. 
        :param init_tau: initial value of membrane potential time constant τ
        :param threshold: threshold for firing
        :param vrest: resting membrane potential
        :paarm reset_mechanism: reset method after firing
        :param spike_grad: approximation function of firing gradient
        :param reset_v: reset membrane potential v (False only when using the last layer's v)
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

        #>> adjust tau to be trainable >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # reference [https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py]
        self.is_train_tau=is_train_tau
        init_w=-log(1/(self.init_tau-min_tau)-1)
        if is_train_tau:
            self.w=nn.Parameter(init_w * torch.ones(size=in_size))  # default initialization
        elif not is_train_tau:
            self.w=(init_w * torch.ones(size=in_size))  # default initialization
        #<< adjust tau to be trainable <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.v=0.0
        self.r=1.0 #membrane resistance


    def forward(self,current:torch.Tensor):
        """
        :param current: synaptic current [batch x ...]
        """

        device=current.device

        if not self.is_train_tau:
            self.w=self.w.to(device)

        # Move v and w to the same device as current if they are not already
        if isinstance(self.v,torch.Tensor):
            if self.v.device != device:
                self.v = self.v.to(device)
        if self.w.device != device:
            self.w = self.w.to(device)


        tau=self.min_tau+self.w.sigmoid() #if tau is too small, dt/tau will exceed 1
        dv=self.dt/(tau) * ( -(self.v-self.vrest) + (self.r)*current ) #increment of membrane potential v
        self.v=self.v+dv
        spike=self.__fire()

        if self.reset_v:
            v_tmp=self.v*(1.0-spike) + self.v_peak*spike #membrane potential before reset
            self.__reset_voltage(spike)
        elif not self.reset_v:
            v_tmp=self.v #do not reset the peak value when reset_v is False

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
    LIF with dynamic time constant
    """

    def __init__(
            self,in_size:tuple,dt,init_tau=0.5,min_tau=0.1,threshold=1.0,
            vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False, 
            reset_v=True, v_actf=None
        ):
        """
        :param in_size: input size of current
        :param dt: ⊿t when the LIF model is converted to a difference equation. If the input is a spike time series, it is the same as the original data. 
        :param init_tau: initial value of membrane potential time constant τ. [[tau1, rate1],[tau2, rate2],...] can be specified to specify the ratio
        :param threshold: threshold for firing
        :param vrest: resting membrane potential. 
        :paarm reset_mechanism: reset method after firing
        :param spike_grad: approximation function of firing gradient
        :param reset_v: reset membrane potential v (False only when using the last layer's v)
        :param v_actf<str>: {None, "relu", "tanh"} activation function of membrane potential v (used to adjust the internal state)
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
        self.v_peak=self.threshold*3.0 #the peak value is set to 3 times the threshold (it is not important for drawing)

        #>> adjust tau to be trainable >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # reference [https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py]
        init_w=-log(1/(self.init_tau-min_tau)-1)
        self.w=nn.Parameter(init_w * torch.ones(size=in_size))  # default initialization
        #<< adjust tau to be trainable <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        self.v=0.0
        self.r=1.0 #membrane resistance
        self.a=1.0 #time scale

        self.__set_v_actf(v_actf) #set activation function of membrane potential v

    def __set_v_actf(self,v_actf):
        if v_actf=="relu":
            self.v_actf=torch.nn.ReLU()
        elif v_actf=="tanh":
            self.v_actf=torch.nn.Tanh()
        else:
            self.v_actf=None


    def forward(self,current:torch.Tensor):
        """
        :param current: synaptic current [batch x ...]
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

        tau=self.min_tau+self.w.sigmoid() #if tau is too small, dt/tau will exceed 1

        tau_new=tau*self.a
        tau_new[tau_new<self.min_tau]=self.min_tau #if tau is too small, dt/tau will exceed 1, so clipping
        
        dv=(self.dt/tau_new) * (-(self.v-self.vrest)) + (self.dt/(tau*self.a)) * (self.a*self.r*current) #clipping only for tau of internal state
       
        self.v=self.v+dv

        if not self.v_actf is None: #if the activation function is set, apply it
            self.v=self.v_actf(self.v)

        spike=self.__fire() 


        if self.reset_v: 
            v_tmp=self.v*(1.0-spike) + self.v_peak*spike #set the peak value of the membrane potential when the spike is fired (for drawing)
            self.__reset_voltage(spike)
        else:
            v_tmp=self.v #if reset_v is False, do not reset the peak value
        
        self._set_state(current,v_tmp,spike) #save the state (for visualization and memory usage gets larger)

        if not self.output:
            return spike
        else:
            return spike, (current,v_tmp) #return spike and (synaptic current, membrane potential)


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
        self.spike_prev=None #reset spike. it is better to use other functions (for readability)
        self.v=0.0

    def _set_state(self,current:torch.Tensor,volt:torch.Tensor,outspike:torch.Tensor):
        """
        save the data of each layer at each time step (for visualization and memory usage gets larger)
        """
        self.lif_state={
            "current":current,
            "volt":volt,
            "outspike":outspike
        }



