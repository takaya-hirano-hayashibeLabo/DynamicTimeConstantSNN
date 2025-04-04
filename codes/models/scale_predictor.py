import torch
import numpy as np
from math import log,exp


class ScalePredictor():
    """
    入力からタイムスケールを予測するクラス
    """
    def __init__(self,datatype="xor"):

        self.datatype=datatype
        self.data_trj=torch.Tensor(np.array([]))


    def predict_scale_trajectory(self,data:torch.Tensor):
        """
        全stepのデータを入力とする
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
        1ステップ分のデータを入力とする
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
        xorの1ステップ分のデータが入る
        :param data: [batch x xdim]
        """

        #>> scaleとfiring rateで線形回帰したときの係数とwindow >>>>>>>>
        window_size=120
        slope,intercept=-1.0437631068421338,-0.6790105922709921 
        #<< scaleとfiring rateで線形回帰したときの係数とwindow <<<<<<<<
        
        self.data_trj=torch.cat([self.data_trj.to(data.device),data.unsqueeze(1)],dim=1)
        if self.data_trj.shape[1]>window_size:
            self.data_trj=self.data_trj[:,(self.data_trj.shape[1]-window_size):] #長くなりすぎたらカット
        
        scale_log=log(self.firing_rate+1e-10)*slope + intercept
        scale=exp(scale_log)

        return scale
    

    def __predict_gesture(self,data:torch.Tensor):
        """
        テスト済み. 想定した通りの挙動をしている
        gestureの1ステップ分のデータが入る
        :param data: [batch x channel x w x h]
        """
        F0=0.06945327669382095 #a=1のときの平均firing rate
        WINDOW_SIZE=300 #frを計算するためのwindow size
        
        # >>> データの追加とpop >>>
        self.data_trj=torch.cat([self.data_trj.to(data.device),data.unsqueeze(1)],dim=1)
        if self.data_trj.shape[1]>WINDOW_SIZE:
            self.data_trj=self.data_trj[:,(self.data_trj.shape[1]-WINDOW_SIZE):] #長くなりすぎたらカット
        # <<<<<<<<<<<<<<<<<<<<<<<<

        # >>> 入力データの発火率を計算 >>>
        fr_in=self.__calc_firing_rate(self.data_trj)
        # <<<<<<<<<<<<<<<<<<<<<<<<

        # >>> スケールを計算 >>>
        scale=F0/fr_in
        # <<<<<<<<<<<<<<<<<<<<<<<<

        return scale


    def __calc_firing_rate(self, data:torch.Tensor)->float:
        """
        :param data: [batch x timestep x channel x w x h]
        """
        fr=torch.mean(data,dim=1) #時間平均
        fr=torch.mean(fr,dim=0) #バッチ平均
        fr=torch.mean(fr) #ピクセル平均
        return fr.item()


    def reset_trj(self):
        self.data_trj=torch.Tensor(np.array([]))
    
    @property
    def firing_rate(self):
        fr=torch.mean(self.data_trj,dim=1) #時間平均
        fr=torch.mean(fr,dim=0) #バッチ平均
        fr=torch.mean(fr) #ピクセル平均
        return fr.item()
