"""
Spike encoder
Gaussian function is used to scale the time step according to the speed change
"""
import torch


class ThresholdEncoder():
    """
    if the input exceeds the threshold, the spike is output
    """
    def __init__(self,thr_max,thr_min,resolution:int,device="cpu"):
        """
        :param thr_max, thr_min: maximum and minimum of threshold
        :param resolution: number of threshold divisions
        """
        self.threshold=torch.linspace(thr_min,thr_max,resolution).to(device)
        self.resolution=resolution
        self.skip_ndim=2 #skip 0,1 dimension (N x T is known)
        self.device=device

    def _is_same_ndim(self,x:torch.Tensor):
        """
        whether the input and threshold have the same number of dimensions
        """
        return x.ndim==self.threshold.ndim
    
    def _reshape_threshold(self,x:torch.Tensor):
        """
        :param x: [N x T x xdim]
        :return: [N x T x resolution x xdim]
        """
        nxdim=x.ndim-self.skip_ndim #number of dimensions after 2nd dimension
        new_thr_size=(1,1,self.resolution)+tuple(1 for _ in range(nxdim))
        
        self.threshold=self.threshold.view(*new_thr_size)

        # xdim_size=tuple(
        #     [x.shape[i+skip_ndim] for i in range(x.ndim-skip_ndim)]
        # ) #number of dimensions after 2nd dimension

    def _get_xdim(self,x:torch.Tensor):
        xdim_size=tuple(
            x.shape[i+self.skip_ndim] for i in range(x.ndim-self.skip_ndim)
        ) #number of dimensions after 2nd dimension
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

        #>> adjust the number of dimensions of threshold according to the input >>
        if not self._is_same_ndim(x): self._reshape_threshold(x)

        #>> shift the input and compare the previous and next values >>
        x_prev=x[:,:-1].unsqueeze(2) #unsqueezeはthreshold次元を追加
        x_next=x[:,1:].unsqueeze(2)

        # >> if the threshold is crossed, True >>
        spike_mask=(
            ((x_prev<=self.threshold) & (self.threshold<x_next)) |
            ((x_next<self.threshold) & (self.threshold<=x_prev))
        )

        out_spike[:,1:]=spike_mask.float().to(self.device)

        return out_spike
        

