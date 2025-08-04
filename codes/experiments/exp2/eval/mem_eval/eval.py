"""
see the firing of each layer
and change the time scale according to it
"""

from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent.parent.parent
import sys
sys.path.append(str(ROOT/"codes"))

import json
import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision
import tonic
import argparse
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
from tqdm import tqdm
from models import DynamicCSNN,CSNN,DynamicResCSNN,ResCSNN,CSNN
from utils import load_yaml


def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # convert numpy.ndarray to Tensor
    inputs = [torch.tensor(input) for input in inputs]
    
    # pad the tensor to the same size
    inputs_padded = pad_sequence(inputs, batch_first=True)
    
    # targets is a list of integers, so convert it to a tensor
    targets_tensor = torch.tensor(targets)
    
    return inputs_padded, targets_tensor



import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

def plot_volt_cmap(volt:torch.Tensor, savepath:Path, filename="volt"):
    """
    :param volt: [timestep x neuron]
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    volt = volt.cpu().detach().numpy()
    plt.figure(figsize=(10, 8))  # Adjusted figure size
    plt.imshow(
        volt.T, cmap="viridis", 
        interpolation="nearest", aspect='auto',
        norm=SymLogNorm(linthresh=1e-3)
    )  # Set aspect to 'auto'
    plt.colorbar()
    plt.title("Voltage Colormap")
    plt.xlabel("Time Step")
    plt.ylabel("Neuron Index")
    plt.tight_layout()  # Automatically adjust subplot parameters
    plt.savefig(savepath / f"{filename}.png")
    plt.close()


def save_volt_csv(volt:torch.Tensor, savepath:Path, filename="volt"):
    """
    :param volt: [timestep x neuron]
    """
    volt = volt.cpu().detach().numpy()
    np.savetxt(savepath / f"{filename}.csv", volt, delimiter=",")

def main():
    is_source_test=False

    parser=argparse.ArgumentParser()
    parser.add_argument("--target",default="")
    parser.add_argument("--device",default=0)
    parser.add_argument("--saveto",default="dynamic-snn")
    parser.add_argument("--batch_head",type=int,default=-1,help="if less than 0, all batches are targeted")
    parser.add_argument("--batch_idx",type=int,default=-1,help="if less than 0, all batches are targeted")
    parser.add_argument("--a",type=float,default=1.0,help="time scale")
    args=parser.parse_args()

    target_batch_head=int(args.batch_head)
    target_batch_idx=int(args.batch_idx)

    modelname=Path(args.target).name

    a=args.a

    relativepath=args.saveto
    resdir=Path(__file__).parent/f"{relativepath}/{modelname}/a{a:.2f}"
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    args_dict=args.__dict__
    json.dump(args_dict,open(resdir/"args.json","w"),indent=4)


    #>> config preparation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf:dict
    train_conf,model_conf=conf["train"],conf["model"]
    model_conf["output-membrane"]=True
    #<< config preparation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> model preparation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    model_conf["memory-lifstate"]=True
    #>> model preparation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="dynamic-snn".casefold():
        if "res".casefold() in model_conf["cnn-type"]:
            model=DynamicResCSNN(model_conf)
        else:
            model=DynamicCSNN(model_conf)
    elif model_conf["type"]=="snn".casefold():
        if "res".casefold() in model_conf["cnn-type"]:
            model=ResCSNN(model_conf)
        else:
            model=CSNN(model_conf)
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    print(model)
    modelname="model_best.pth"
    model.load_state_dict(torch.load(Path(args.target)/f"result/{modelname}",map_location=device))
    model.to(device)
    #<< model preparation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    model.eval()

    # Debugging parameters of each layer
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print first 2 values for brevity    model.eval(
    is_train_data=False
    datapath=ROOT/"data/original-data"
    time_window=3000
    insize=model_conf["in-size"]
    batch_size=16
    time_sequence=300
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=time_window),
        torch.from_numpy,
    ])

    testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=is_train_data,transform=transform)
    cachepath=ROOT/f"data/cache-data/gesture/window{time_window}-insize{insize}/test"
    if os.path.exists(cachepath):
        testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath))
    testloader=DataLoader(testset,batch_size=batch_size,shuffle=False,collate_fn=custom_collate_fn,num_workers=1)


    print(f"rollout base inputs...")
    predicted_labels_base=[]
    for batch_head, (base_in, targets) in tqdm(enumerate(testloader)):

        if target_batch_head>=0 and batch_head!=target_batch_head: continue #skip if not the specified batch head

        base_in=base_in[:,:time_sequence]
        base_in_count=base_in.clone()
        base_in[base_in>0]=1.0 #spike clip
        base_in=torch.Tensor(base_in).permute(1,0,2,3,4)

        with torch.no_grad():
            base_s,_,base_v=model.forward(base_in.to(device).to(torch.float))

        for batch_i in range(base_s.shape[1]):

            if target_batch_idx>=0 and batch_i!=target_batch_idx: continue #skip if not the specified batch index

            # print(f"base_in spike counts: {base_in[:,batch_i].sum()}")
            # print(f"base_v shape: {base_v[:,batch_i].shape}")
            plot_volt_cmap(base_v[:,batch_i],resdir/f"batch-head{batch_head}_batch-idx{batch_i}",filename=f"base_v")
            save_volt_csv(base_v[:,batch_i],resdir/f"batch-head{batch_head}_batch-idx{batch_i}",filename=f"base_v")
            predicted_labels_base.append(torch.argmax(torch.sum(base_s,dim=0),dim=1)[batch_i].item())
        if is_source_test:
            break
    print(f"done.")


    #>> real #>>>>>>>>>>>>>>>>>
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=int(time_window/a)),
        torch.from_numpy,
    ])
    testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=is_train_data,transform=transform)
    cachepath=ROOT/f"data/cache-data/gesture/window{int(time_window/a)}-insize{insize}/test"
    if os.path.exists(cachepath):
        testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath))
    testloader_scaled=DataLoader(testset,batch_size=batch_size,shuffle=False,collate_fn=custom_collate_fn,num_workers=1)


    print(f"rollout scaled inputs...")
    batch_head_list=[]
    batch_idx_list=[]
    true_labels=[]
    predicted_labels=[]
    for batch_head, (scaled_in, targets) in tqdm(enumerate(testloader_scaled)):

        if target_batch_head>=0 and batch_head!=target_batch_head: continue #skip if not the specified batch head

        scaled_in=scaled_in[:,:int(a*time_sequence)]
        scaled_in_count=scaled_in.clone()
        scaled_in[scaled_in>0]=1.0 #spike clip
        scaled_in=torch.Tensor(scaled_in).permute(1,0,2,3,4)
        print(f"scaled_in spike counts: {scaled_in.sum()}")

        with torch.no_grad():
            if "dynamic".casefold() in model_conf["type"].casefold():
                scaled_s,_,scaled_v=model.dynamic_forward_v1(
                    scaled_in.to(device).to(torch.float),
                    a=torch.Tensor([a for _ in range(scaled_in.shape[0])])
                    )
            else:
                scaled_s,_,scaled_v=model.forward(scaled_in.to(device).to(torch.float))

        for batch_i in range(scaled_s.shape[1]):

            if target_batch_idx>=0 and batch_i!=target_batch_idx: continue #skip if not the specified batch index

            # print(f"scaled_in spike counts: {scaled_in[:,batch_i].sum()}")
            # print(f"scaled_v shape: {scaled_v[:,batch_i].shape}")
            plot_volt_cmap(scaled_v[:,batch_i],resdir/f"batch-head{batch_head}_batch-idx{batch_i}",filename=f"scaled_v_a{a:.2f}")
            save_volt_csv(scaled_v[:,batch_i],resdir/f"batch-head{batch_head}_batch-idx{batch_i}",filename=f"scaled_v_a{a:.2f}")
            batch_head_list.append(batch_head)
            batch_idx_list.append(batch_i)
            true_labels.append(targets[batch_i].item())
            predicted_labels.append(torch.argmax(torch.sum(scaled_s,dim=0),dim=1)[batch_i].item())

        print(f"targets: {targets.cpu().numpy()}")
        print(f"predicted: {torch.argmax(torch.sum(scaled_s,dim=0),dim=1).cpu().numpy()}")

        if is_source_test:
            break
    print(f"done.")
    import pandas as pd
    result_df=pd.DataFrame({
        "batch_head":batch_head_list,
        "batch_idx":batch_idx_list,
        "true_label":true_labels,
        "predicted_label_base":predicted_labels_base,
        "predicted_label_scaled":predicted_labels,
    })
    result_df.to_csv(resdir/"result.csv",index=False)
    print(result_df)

if __name__=="__main__":
    main()