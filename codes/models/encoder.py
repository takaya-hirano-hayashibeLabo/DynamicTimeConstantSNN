"""
スパイクエンコーダ
速度変化に応じてガウス関数のタイムスケールが上手く起きるようにしたもの
"""
import torch


class ThresholdEncoder():
    """
    閾値を超えたらスパイクを出力する
    """
    def __init__(self,thr_max,thr_min,resolution:int,device="cpu"):
        """
        :param thr_max, thr_min: 閾値の最大, 最小
        :param resolution: 閾値の分割数
        """
        self.threshold=torch.linspace(thr_min,thr_max,resolution).to(device)
        self.resolution=resolution
        self.skip_ndim=2 #0,1次元目は飛ばす(N x Tとわかっているため)
        self.device=device

    def _is_same_ndim(self,x:torch.Tensor):
        """
        入力とthresholdの次元数が一致しているか
        """
        return x.ndim==self.threshold.ndim
    
    def _reshape_threshold(self,x:torch.Tensor):
        """
        :param x: [N x T x xdim]
        :return: [N x T x resolution x xdim]
        """
        nxdim=x.ndim-self.skip_ndim #2次元目以降の次元数
        new_thr_size=(1,1,self.resolution)+tuple(1 for _ in range(nxdim))
        
        self.threshold=self.threshold.view(*new_thr_size)

        # xdim_size=tuple(
        #     [x.shape[i+skip_ndim] for i in range(x.ndim-skip_ndim)]
        # ) #2次元目以降の次元サイズ

    def _get_xdim(self,x:torch.Tensor):
        xdim_size=tuple(
            x.shape[i+self.skip_ndim] for i in range(x.ndim-self.skip_ndim)
        ) #2次元目以降の次元サイズ
        return xdim_size
    
    def __call__(self,x:torch.Tensor):
        """
        :param x: [N x T x xdim]
        :return: [N x T x resolution x xdim]
        """
        if not isinstance(x, torch.Tensor):
            x=torch.Tensor(x)

        N, T = x.shape[0],x.shape[1]

        out_spike_shape=(N,T,self.resolution) + self._get_xdim(x)
        out_spike=torch.zeros(out_spike_shape,device=self.device)

        #>> 入力に合わせてthresholdの次元数を調整 >>
        if not self._is_same_ndim(x): self._reshape_threshold(x)

        #>> 入力をシフトして前後の値を比較 >>
        x_prev=x[:,:-1].unsqueeze(2) #unsqueezeはthreshold次元を追加
        x_next=x[:,1:].unsqueeze(2)

        # >> 閾値をまたいだらTrueとする >>
        spike_mask=(
            ((x_prev<=self.threshold) & (self.threshold<x_next)) |
            ((x_next<self.threshold) & (self.threshold<=x_prev))
        )

        out_spike[:,1:]=spike_mask.float().to(self.device)

        return out_spike
        

