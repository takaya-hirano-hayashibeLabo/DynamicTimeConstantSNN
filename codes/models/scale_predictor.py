import torch
import numpy as np
from math import log,exp


class ScalePredictor():
    """
    class to predict the time scale from the input
    """
    def __init__(self,datatype="xor"):

        self.datatype=datatype
        self.data_trj=torch.Tensor(np.array([]))


    def predict_scale_trajectory(self,data:torch.Tensor):
        """
        input is all step data
        :param data: [timestep x batch x x_dim]
        :return: [timestep x batch]
        """
        scale_trj=[]
        for t in range(data.shape[0]):
            timescale_t=self.predict_scale(data[t])
            scale_trj.append(timescale_t)
        scale_trj=torch.Tensor(np.array(scale_trj))

        self.reset_trj()

        return scale_trj


    def predict_scale(self,data:torch.Tensor):
        """
        input is 1 step data
        :param data: [batch x x_dim]
        """

        scale=1
        if self.datatype=="xor":
            scale=self.__predict_xor(data)
        if self.datatype=="gesture":
            scale=self.__predict_gesture(data)

        return scale


    def __predict_xor(self,data:torch.Tensor):
        """
        input is 1 step data of xor
        :param data: [batch x xdim]
        """

        #>> coefficients and window size when linear regression with scale and firing rate >>>>>>>>
        window_size=120
        slope,intercept=-1.0437631068421338,-0.6790105922709921 
        #<< coefficients and window size when linear regression with scale and firing rate <<<<<<<<
        
        self.data_trj=torch.cat([self.data_trj.to(data.device),data.unsqueeze(1)],dim=1)
        if self.data_trj.shape[1]>window_size:
            self.data_trj=self.data_trj[:,(self.data_trj.shape[1]-window_size):] #if too long, cut
        
        scale_log=log(self.firing_rate+1e-10)*slope + intercept
        scale=exp(scale_log)

        return scale
    

    def __predict_gesture(self,data:torch.Tensor):
        """
        tested. it behaves as expected
        input is 1 step data of gesture
        :param data: [batch x channel x w x h]
        """
        F0=0.06945327669382095 #average firing rate when a=1
        WINDOW_SIZE=300 #window size for calculating firing rate
        
        # >>> add data and pop >>>
        self.data_trj=torch.cat([self.data_trj.to(data.device),data.unsqueeze(1)],dim=1)
        if self.data_trj.shape[1]>WINDOW_SIZE:
            self.data_trj=self.data_trj[:,(self.data_trj.shape[1]-WINDOW_SIZE):] #if too long, cut
        # <<<<<<<<<<<<<<<<<<<<<<<<

        # >>> calculate firing rate of input data >>>
        fr_in=self.__calc_firing_rate(self.data_trj)
        # <<<<<<<<<<<<<<<<<<<<<<<<

        # >>> calculate scale >>>
        scale=F0/fr_in
        # <<<<<<<<<<<<<<<<<<<<<<<<

        return scale


    def __calc_firing_rate(self, data:torch.Tensor)->float:
        """
        :param data: [batch x timestep x channel x w x h]
        """
        fr=torch.mean(data,dim=1) #time average
        fr=torch.mean(fr,dim=0) #batch average
        fr=torch.mean(fr) #pixel average
        return fr.item()


    def reset_trj(self):
        self.data_trj=torch.Tensor(np.array([]))
    
    @property
    def firing_rate(self):
        fr=torch.mean(self.data_trj,dim=1) #time average
        fr=torch.mean(fr,dim=0) #batch average
        fr=torch.mean(fr) #pixel average
        return fr.item()
