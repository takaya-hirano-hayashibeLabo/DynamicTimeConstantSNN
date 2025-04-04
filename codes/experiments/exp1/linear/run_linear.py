from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
PARENT=Path(__file__).parent
import sys
sys.path.append(str(ROOT))
print(ROOT)

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

from models import DynamicSNN
from utils import load_yaml,print_terminal


def save_voltage(v_scaled, v_snn, v_dyna, savepath):
    """
    :param v_scaled: [T x 1]
    """

    fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(10,5))
    axes[0].plot(v_scaled, label="Base")
    axes[0].legend()
    axes[1].set_ylim(axes[0].get_ylim())  # ax0のy軸の範囲をax1に設定
    axes[1].plot(v_snn,label="SNN")
    axes[1].plot(v_dyna,label="DynaSNN")
    axes[1].legend()
    plt.savefig(savepath/f"voltage.png")
    plt.close()


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

            track_data=pd.DataFrame(
                data=np.concatenate([
                    timesteps.reshape(-1,1),
                    input_base[:,i_batch],
                    s_base[:,i_batch],
                    i_base[:,i_batch],
                    v_base[:,i_batch],
                    input_scaled[:,i_batch],
                    v_scaled[:,i_batch],
                    s_snn[:,i_batch],
                    i_snn[:,i_batch],
                    v_snn[:,i_batch],
                    s_dyna[:,i_batch],
                    i_dyna[:,i_batch],
                    v_dyna[:,i_batch],
                ],axis=1),
                columns=[
                    "timesteps",
                    "input_base","s_base","i_base","v_base",
                    "input_scaled","v_scaled",
                    "s_snn","i_snn","v_snn",
                    "s_dyna","i_dyna","v_dyna"
                ]
            )
            track_data.to_csv(track_datapath/f"trackdata.csv",index=False)
            save_voltage(v_scaled[:,i_batch],v_snn[:,i_batch],v_dyna[:,i_batch],track_datapath)

        # print(f"timesteps: {timesteps.shape}\ninput_base: {input_base.shape}\ns_base: {s_base.shape}\ni_base: {i_base.shape}\nv_base: {v_base.shape}\nv_scaled: {v_scaled.shape}\ninput_scaled: {input_scaled.shape}\ns_snn: {s_snn.shape}\ni_snn: {i_snn.shape}\nv_snn: {v_snn.shape}\ns_dyna: {s_dyna.shape}\ni_dyna: {i_dyna.shape}\nv_dyna: {v_dyna.shape}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--testnums",type=int,default=10)
    parser.add_argument("--timescale",type=float,default=3)
    parser.add_argument("--tau",type=float,default=0.008)
    parser.add_argument("--batchsize",type=int,default=1000)
    parser.add_argument("--track_batchsize",type=int, default=100,help="データをcsvで保存するバッチサイズ")
    parser.add_argument("--device",default=0)
    args=parser.parse_args()

    track_batchsize=args.track_batchsize

    result_path=PARENT/f"result_tau{args.tau:.3f}/timescale{args.timescale:.2f}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    track_datapath=result_path/"track_data"
    if not os.path.exists(track_datapath):
        os.makedirs(track_datapath)

    device=torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    config=load_yaml(PARENT/"config.yml") #DynaSNNの設定
    config["model"]["init-tau"]=float(args.tau) #時定数を変える
    print_terminal(f"running processing [timescale: {args.timescale:.2f}]...")

    
    result_trajectory=[]
    mse_table=[]
    for i_test in tqdm(range(args.testnums)):
        model=DynamicSNN(conf=config["model"]).to(device) # テストエポックごとにモデルを初期化する. なぜなら未学習のモデルの出力は初期重みに死ぬほど依存するから

        T=300
        batch=args.batchsize #バッチよりもモデルの初期値に依存する
        insize=config["model"]["in-size"]

        p=0.1
        base_input=torch.where(
            torch.rand(size=(T,batch,insize))<p,1.0,0.0
        ).to(device)
        base_s,base_i,base_v=model(base_input)

        a=args.timescale  # 'a' can now be a float
        # Create scaled_input by shifting indices by a factor of 'a'
        scaled_input = torch.zeros(size=(int(a * T), batch, insize)).to(device)
        if a >= 1.0:
            kernel_size=a
            for t in range(T):
                scaled_index = int(a * t)
                if scaled_index < scaled_input.shape[0]:
                    scaled_input[scaled_index] = base_input[t]
        else:
            # 畳み込みを行う処理
            kernel_size = math.ceil(1 / a)  # カーネルサイズを設定
            
            # print(f"base_input.shape: {base_input.shape}")
            scaled_input=torch.Tensor([]).to(base_input.device)
            for dim_i in range(config["model"]["in-size"]):
                scaled_input_i = F.conv1d(base_input.permute(1, 2, 0)[:,dim_i].unsqueeze(1), 
                                                weight=torch.ones(1, 1, kernel_size).to(base_input.device), 
                                                stride=kernel_size).permute(2, 0, 1)
                scaled_input_i[scaled_input_i<0.5]=0.0
                scaled_input_i[scaled_input_i>=0.5]=1.0
                # print(f"scaled_input_i.shape: {scaled_input_i.shape}")
                scaled_input=torch.cat([scaled_input,scaled_input_i],dim=-1)
            # scaled_input=torch.stack(scaled_input,dim=-1).to(base_input.device).squeeze(-2)
            # print(f"scaled_input.shape: {scaled_input.shape}")

        org_s,org_i,org_v=model.forward(scaled_input)
        scaled_s,scaled_i,scaled_v=model.dynamic_forward_v1(scaled_input,a=torch.Tensor([a for _ in range(scaled_input.shape[0])]))
        
        
        v1_resampled=F.interpolate(base_v.permute(1,2,0), size=int(a*T), mode='linear', align_corners=False).permute(-1,0,1) #基準膜電位のタイムスケール(線形補間)
        
        scaled_T=scaled_input.shape[0]

        mse_snn_arr=np.mean((v1_resampled.to("cpu").detach().numpy()[:scaled_T]-org_v.to("cpu").detach().numpy())**2,axis=0) #時間方向に平均をとる
        mse_snn=  np.mean(mse_snn_arr) 
        mse_dyna_arr=np.mean((v1_resampled.to("cpu").detach().numpy()[:scaled_T]-scaled_v.to("cpu").detach().numpy())**2,axis=0)
        mse_dyna= np.mean(mse_dyna_arr) 
        result_trajectory.append((mse_snn,mse_dyna))

        mse_table+=np.concatenate([
            np.ones(shape=(track_batchsize,1))*i_test, #モデル番号
            np.arange(track_batchsize).reshape(-1,1), #バッチ番号
            mse_snn_arr[:track_batchsize].mean(axis=-1).reshape(-1,1), #SNNのMSE
            mse_dyna_arr[:track_batchsize].mean(axis=-1).reshape(-1,1), #DynaSNNのMSE
        ],axis=1).tolist()
        mse_table_pd=pd.DataFrame(mse_table,columns=["model idx","batch idx","mse_snn","mse_dyna"])
        mse_table_pd.to_csv(track_datapath/f"mse_table.csv",index=False)
        

        # データをトラッキング
        save_tarck_data(
            timesteps=torch.arange(0,scaled_v.shape[0]).cpu().numpy(),
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
            track_batchsize=args.track_batchsize,
            savepath=track_datapath,
        )


    result_trajectory=np.array(result_trajectory)
    result_dict={
        "timescale":args.timescale,
        "testnums":args.testnums,
        "lif_mean":np.mean(result_trajectory[:,0]).astype(float),
        "lif_std":np.std(result_trajectory[:,0]).astype(float),
        "dyna_mean":np.mean(result_trajectory[:,1]).astype(float),
        "dyna_std":np.std(result_trajectory[:,1]).astype(float),
    }


    # 引数をjsonに保存
    args_dict=vars(args)
    json.dump(args_dict,open(result_path/"args.json",'w'),indent=4)


    with open(result_path/f"result.json",'w') as f:
        json.dump(result_dict,f,indent=4)


if __name__ == "__main__":
    main()