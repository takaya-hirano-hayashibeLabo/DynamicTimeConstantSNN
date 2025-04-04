from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))
print(ROOT)
PARENT=Path(__file__).parent

import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import argparse
import numpy as np
import json
import os
import math
from tqdm import tqdm
from torch import nn

from models import DynamicCSNN,DynamicResCSNN
from utils import load_yaml,print_terminal

def plot_voltage_heatmap(v_scaled:np.ndarray,v_snn:np.ndarray,v_dyna:np.ndarray,savepath:Path):
    fig,axes=plt.subplots(nrows=3,ncols=1,figsize=(10,8))
    axes[0].imshow(v_scaled.T,cmap="viridis", aspect="auto" ,interpolation="nearest")
    axes[0].set_title("Base Voltage")
    axes[1].imshow(v_snn.T,cmap="viridis", aspect="auto" ,interpolation="nearest")
    axes[1].set_title("SNN Voltage")
    axes[2].imshow(v_dyna.T,cmap="viridis", aspect="auto" ,interpolation="nearest")
    axes[2].set_title("Dyna Voltage")
    plt.tight_layout()
    plt.savefig(savepath/f"voltage_heatmap.png")
    plt.close()

def save_heatmap2csv(array:np.ndarray,column_prefix:str,savepath:Path):
    """
    :param array: [T x xdim...]
    """

    T=array.shape[0]
    heatmap=array.reshape(T,-1)
    xdim=heatmap.shape[1]
    heatmap_pd=pd.DataFrame(
        data=heatmap,
        columns=[f"{column_prefix}{i}" for i in range(xdim)]
    )
    heatmap_pd.to_csv(savepath,index=False)

def save_tarck_data(
        timesteps, input_base, s_base, i_base, v_base, v_scaled,
        input_scaled, 
        s_snn, i_snn, v_snn,
        s_dyna, i_dyna, v_dyna,
        test_idx, track_batchsize, savepath):

        #-- データのパディング ------------------------------------------------------------
        T_base=input_base.shape[0]
        T_scaled=v_scaled.shape[0]

        def _pad_array(array:np.ndarray, dT:int, shape:tuple):
            new_array=np.concatenate(
                [array,np.full(shape=(dT, *shape),fill_value=np.nan)],axis=0
            )
            return new_array

        if T_base>T_scaled:
            dT=T_base-T_scaled
            timesteps=np.arange(0,T_base)
            v_scaled=_pad_array(v_scaled,dT,v_scaled.shape[1:])
            input_scaled=_pad_array(input_scaled,dT,input_scaled.shape[1:])
            s_snn=_pad_array(s_snn,dT,s_snn.shape[1:])
            i_snn=_pad_array(i_snn,dT,i_snn.shape[1:])
            v_snn=_pad_array(v_snn,dT,v_snn.shape[1:])
            s_dyna=_pad_array(s_dyna,dT,s_dyna.shape[1:])
            i_dyna=_pad_array(i_dyna,dT,i_dyna.shape[1:])
            v_dyna=_pad_array(v_dyna,dT,v_dyna.shape[1:])

        elif T_base<T_scaled:
            dT=T_scaled-T_base
            input_base=_pad_array(input_base,dT,input_base.shape[1:])
            s_base=_pad_array(s_base,dT,s_base.shape[1:])
            i_base=_pad_array(i_base,dT,i_base.shape[1:])
            v_base=_pad_array(v_base,dT,v_base.shape[1:])


        
        #-- データをcsv形式にして保存 ------------------------------------------------------------
        for i_batch in (range(track_batchsize)):
            track_datapath=savepath/f"model{test_idx}/batch{i_batch}"
            if not os.path.exists(track_datapath):
                os.makedirs(track_datapath)
            
            save_heatmap2csv(input_base[:,i_batch],column_prefix="input_base",savepath=track_datapath/"input_base.csv")
            save_heatmap2csv(s_base[:,i_batch],column_prefix="s_base",savepath=track_datapath/"s_base.csv")
            save_heatmap2csv(i_base[:,i_batch],column_prefix="i_base",savepath=track_datapath/"i_base.csv")
            save_heatmap2csv(v_base[:,i_batch],column_prefix="v_base",savepath=track_datapath/"v_base.csv")
            save_heatmap2csv(v_scaled[:,i_batch],column_prefix="v_scaled",savepath=track_datapath/"v_scaled.csv")
            save_heatmap2csv(input_scaled[:,i_batch],column_prefix="input_scaled",savepath=track_datapath/"input_scaled.csv")
            save_heatmap2csv(s_snn[:,i_batch],column_prefix="s_snn",savepath=track_datapath/"s_snn.csv")
            save_heatmap2csv(i_snn[:,i_batch],column_prefix="i_snn",savepath=track_datapath/"i_snn.csv")
            save_heatmap2csv(v_snn[:,i_batch],column_prefix="v_snn",savepath=track_datapath/"v_snn.csv")
            save_heatmap2csv(s_dyna[:,i_batch],column_prefix="s_dyna",savepath=track_datapath/"s_dyna.csv")
            save_heatmap2csv(i_dyna[:,i_batch],column_prefix="i_dyna",savepath=track_datapath/"i_dyna.csv")
            save_heatmap2csv(v_dyna[:,i_batch],column_prefix="v_dyna",savepath=track_datapath/"v_dyna.csv")

            plot_voltage_heatmap(v_scaled[:,i_batch],v_snn[:,i_batch],v_dyna[:,i_batch],savepath=track_datapath)


        # print(f"timesteps: {timesteps.shape}\ninput_base: {input_base.shape}\ns_base: {s_base.shape}\ni_base: {i_base.shape}\nv_base: {v_base.shape}\nv_scaled: {v_scaled.shape}\ninput_scaled: {input_scaled.shape}\ns_snn: {s_snn.shape}\ni_snn: {i_snn.shape}\nv_snn: {v_snn.shape}\ns_dyna: {s_dyna.shape}\ni_dyna: {i_dyna.shape}\nv_dyna: {v_dyna.shape}")


def plot_and_save_inputs(base_input:torch.Tensor, scaled_input:torch.Tensor, save_path:Path, batch_index:int=0,window:int=10):

    base_in_plot=base_input[:,batch_index].flatten(start_dim=1).cpu().numpy()
    scaled_in_plot=scaled_input[:,batch_index].flatten(start_dim=1).cpu().numpy()

    xdim=base_input.shape[-1]

    plt.figure(figsize=(12, 6))
    
    # Base Inputの描画
    plt.subplot(2, 1, 1)
    plt.title("Base Input")
    for i in range(xdim):
        plt.plot(base_in_plot[:,i]*(i+1),"o")
    plt.ylim(0.5,xdim+0.5)
    plt.xticks(ticks=np.arange(0, base_input.shape[0], window)-0.5)  # x軸の間隔をwindowに設定
    plt.grid()
    # print(base_input[:,batch_index].flatten().cpu().numpy())
    
    # Scaled Inputの描画
    plt.subplot(2, 1, 2)
    plt.title("Scaled Input")
    for i in range(xdim):
        plt.plot(scaled_in_plot[:,i]*(i+1),"o")
    plt.ylim(0.5,xdim+0.5)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def memory_batchnrm_params(model:nn.Module, in_size, in_channel,T, p, minibatchsize, epoch, device):
    """
    BatchNorm構造を持つモデルに学習データを流す. これによって平均と分散を記憶させる
    :param model: モデル
    :param in_size: 入力サイズ
    :param in_channel: 入力チャンネル数
    :param T: 時間ステップ数
    :param p: スパイク確率
    :param minibatchsize: ミニバッチサイズ
    :param epoch: エポック数
    """

    model.train() #trainモードにしてbatchnrmのパラメータを更新させる
    for e in range(epoch):
        spikes=torch.where(
            torch.rand(size=(T, minibatchsize, in_channel,in_size,in_size))<p,1.0,0.0
        ).to(device)
        _,_,_=model(spikes)


def debug_batchnorm_params(model: nn.Module):
    """
    モデル内のBatchNorm層の平均（μ）と分散（σ）をデバッグする関数
    :param model: PyTorchのモデル
    """
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            print(f"Layer: {name}")
            print(f"  Running Mean (μ): {layer.running_mean.data}")
            print(f"  Running Variance (σ): {layer.running_var.data}")
            # print(f"  Weight: {layer.weight.data}")
            # print(f"  Bias: {layer.bias.data}")
            print("-" * 40)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--testnums",type=int,default=10)
    parser.add_argument("--timescale",type=float,default=1.0)
    parser.add_argument("--tau",type=float,default=0.008)
    parser.add_argument("--batchsize",type=int,default=10)
    parser.add_argument("--track_batchsize",type=int, default=5,help="データをcsvで保存するバッチサイズ")
    parser.add_argument("--device",default=0)
    parser.add_argument("--modeltype",default="csnn",help="csnn, csnn_dropout, rescsnn")
    # parser.add_argument("--saveto",default="")
    # parser.add_argument("--confpath",default="")
    args=parser.parse_args()

    track_batchsize=args.track_batchsize

    result_path=PARENT/f"results/{args.modeltype}/result_tau{args.tau:.3f}/timescale{args.timescale:.2f}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    track_datapath=result_path/"track_data"
    if not os.path.exists(track_datapath):
        os.makedirs(track_datapath)

    device=torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    config=load_yaml(str(PARENT/f"configs/{args.modeltype}.yml")) #DynaSNNの設定
    print_terminal(f"running processing [timescale: {args.timescale:.2f}]...")

    
    result_trajectory=[]
    mse_table=[]
    for i_test in tqdm(range(args.testnums)):

        if "cnn-type" in config["model"] and config["model"]["cnn-type"]=="res":
            model=DynamicResCSNN(conf=config["model"]).to(device) # テストエポックごとにモデルを初期化する. なぜなら未学習のモデルの出力は初期重みに死ぬほど依存するから
        else:
            model=DynamicCSNN(conf=config["model"]).to(device) # テストエポックごとにモデルを初期化する. なぜなら未学習のモデルの出力は初期重みに死ぬほど依存するから
        
        T=300
        batch=args.batchsize #バッチよりもモデルの初期値に依存する
        insize=config["model"]["in-size"]
        in_channel=config["model"]["in-channel"]
        minibatch=50 if batch>50 else batch
        p=0.1 #スパイク確率


        if config["model"]["is-bn"]: #BatchNorm構造を持つモデルの場合はパラメータを記憶させる
            memory_batchnrm_params(
                model, config["model"]["in-size"], config["model"]["in-channel"], 
                T, p, minibatch, 10, device
            )
        model.eval() #モデルを評価モードにする


        squared_error_lif_minibatch=[]
        squared_error_dyna_minibatch=[]
        for i_minibatch in tqdm(range(math.ceil(batch/minibatch))):
            base_input=torch.where(
                torch.rand(size=(T, minibatch, in_channel,insize,insize))<p,1.0,0.0
            ).to(device)
            base_s,base_i,base_v=model(base_input)

            a=args.timescale  # 'a' can now be a float
            # Create scaled_input by shifting indices by a factor of 'a'
            scaled_input = torch.zeros(size=(int(a * T), minibatch, in_channel,insize,insize)).to(device)
            if a >= 1.0:
                kernel_size=a
                for t in range(T):
                    scaled_index = int(a * t)
                    if scaled_index < scaled_input.shape[0]:
                        scaled_input[scaled_index] = base_input[t]
            else:
                # 1次元の畳み込みを時間方向に行う処理
                kernel_size = math.ceil(1 / a)  # カーネルサイズを設定
                
                # Permute base_input to bring the time dimension to the last position
                base_input_1d = base_input.permute(1, 2, 3, 4, 0)  # (batch, channel, w, h, T)
                
                # Reshape for 1D convolution
                reshaped_input = base_input_1d.view(minibatch * in_channel * insize * insize, T)
                
                # Create a weight tensor with the same number of input channels
                weight = torch.ones(1, 1, kernel_size).to(base_input.device)
                
                # Apply 1D convolution along the time dimension
                scaled_input = F.conv1d(reshaped_input.unsqueeze(1), 
                                        weight=weight, 
                                        stride=kernel_size).view(minibatch, in_channel, insize, insize, -1).permute(4, 0, 1, 2, 3)
                
                scaled_input[scaled_input<0.5]=0.0
                scaled_input[scaled_input>=0.5]=1.0

                # print(scaled_input.shape)

            org_s,org_i,org_v=model.forward(scaled_input)
            scaled_s,scaled_i,scaled_v=model.dynamic_forward_v1(scaled_input,a=torch.Tensor([a for _ in range(scaled_input.shape[0])]))
            
            
            v1_resampled=F.interpolate(base_v.permute(1,2,0), size=int(a*T), mode='linear', align_corners=False).permute(-1,0,1) #基準膜電位のタイムスケール(線形補間)
            
            scaled_T=scaled_input.shape[0]
            squared_error_lif_minibatch+=list(v1_resampled.to("cpu").detach().numpy()[:scaled_T]-org_v.to("cpu").detach().numpy())
            squared_error_dyna_minibatch+=list(v1_resampled.to("cpu").detach().numpy()[:scaled_T]-scaled_v.to("cpu").detach().numpy())


            #-- trackするデータを保存 ------------------------------------------------------------
            mse_snn_arr=np.mean((v1_resampled.to("cpu").detach().numpy()[:scaled_T]-org_v.to("cpu").detach().numpy())**2,axis=0) #時間方向に平均をとる
            mse_dyna_arr=np.mean((v1_resampled.to("cpu").detach().numpy()[:scaled_T]-scaled_v.to("cpu").detach().numpy())**2,axis=0)
            mse_table+=np.concatenate([
                np.ones(shape=(track_batchsize,1))*i_test, #モデル番号
                np.arange(track_batchsize).reshape(-1,1), #バッチ番号
                mse_snn_arr.mean(axis=-1)[:track_batchsize].reshape(-1,1), #SNNのMSE
                mse_dyna_arr.mean(axis=-1)[:track_batchsize].reshape(-1,1), #DynaSNNのMSE
            ],axis=1).tolist()
            mse_table_pd=pd.DataFrame(mse_table,columns=["model idx","batch idx","mse_snn","mse_dyna"])
            mse_table_pd.to_csv(track_datapath/f"mse_table.csv",index=False)
            
            if i_minibatch==0: #最初のバッチだけトラックする
                save_tarck_data(
                    timesteps=np.arange(0,scaled_v.shape[0]),
                    input_base=base_input.to("cpu").detach().numpy(),
                    s_base=base_s.to("cpu").detach().numpy(),
                    i_base=base_i.to("cpu").detach().numpy(),
                    v_base=base_v.to("cpu").detach().numpy(),
                    v_scaled=v1_resampled.to("cpu").detach().numpy()[:scaled_T],
                    input_scaled=scaled_input.to("cpu").detach().numpy(),
                    s_snn=org_s.to("cpu").detach().numpy(),
                    i_snn=org_i.to("cpu").detach().numpy(),
                    v_snn=org_v.to("cpu").detach().numpy(),
                    s_dyna=scaled_s.to("cpu").detach().numpy(),
                    i_dyna=scaled_i.to("cpu").detach().numpy(),
                    v_dyna=scaled_v.to("cpu").detach().numpy(),
                    test_idx=i_test,
                    track_batchsize=track_batchsize,
                    savepath=track_datapath
                )
            #-- trackするデータを保存 ------------------------------------------------------------

        mse_lif=np.mean(np.array(squared_error_lif_minibatch)**2)
        mse_dyna=np.mean(np.array(squared_error_dyna_minibatch)**2)
        result_trajectory.append((mse_lif,mse_dyna))

    result_trajectory=np.array(result_trajectory)
    result_dict={
        "timescale":args.timescale,
        "testnums":args.testnums,
        "lif_mean":np.mean(result_trajectory[:,0]).astype(float),
        "lif_std":np.std(result_trajectory[:,0]).astype(float),
        "dyna_mean":np.mean(result_trajectory[:,1]).astype(float),
        "dyna_std":np.std(result_trajectory[:,1]).astype(float),
    }


    args_dict=vars(args)
    json.dump(args_dict,open(result_path/"args.json",'w'),indent=4)
    json.dump(result_dict,open(result_path/"result.json",'w'),indent=4)

    # inspike_img_path=result_path/"imgs/inspike/"
    # if not os.path.exists(inspike_img_path):
    #     os.makedirs(inspike_img_path)
    # plot_and_save_inputs(base_input,scaled_input,inspike_img_path/f"inspikes_timescale{args.timescale:.2f}.png",batch_index=0,window=kernel_size)
    # # print(base_input)

if __name__ == "__main__":
    main()