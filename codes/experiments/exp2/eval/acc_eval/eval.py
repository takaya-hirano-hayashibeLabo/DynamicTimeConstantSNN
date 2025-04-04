import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent.parent.parent
EXP=Path(__file__).parent
import sys
sys.path.append(str(ROOT/"codes"))

import os
import torch
import json
import numpy as np
from tqdm import tqdm
from snntorch import functional as SF
from torch.utils.data import DataLoader
import pandas as pd
import torchvision
import tonic
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from math import floor
import random
import re


from utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json,save_heatmap_video
from models import DynamicCSNN,CSNN,DynamicResCSNN,ResNetLSTM,ResCSNN,ScalePredictor


def plot_and_save_curves(result, resultpath, epoch):
    df = pd.DataFrame(result, columns=["epoch", "train_loss_mean", "train_loss_std", "train_acc_mean", "train_acc_std", "val_loss_mean", "val_loss_std", "val_acc_mean", "val_acc_std"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    axes[0].errorbar(df['epoch'], df['train_loss_mean'], yerr=df['train_loss_std'], label='Train Loss')
    axes[0].errorbar(df['epoch'], df['val_loss_mean'], yerr=df['val_loss_std'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curve')

    # Plot Accuracy
    axes[1].errorbar(df['epoch'], df['train_acc_mean'], yerr=df['train_acc_std'], label='Train Accuracy')
    axes[1].errorbar(df['epoch'], df['val_acc_mean'], yerr=df['val_acc_std'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig(resultpath / f'train_curves.png')
    plt.close()



def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # numpy.ndarrayをTensorに変換
    inputs = [torch.tensor(input) for input in inputs]
    
    # パディングを使用してテンソルのサイズを揃える
    inputs_padded = pad_sequence(inputs, batch_first=True)
    
    # targetsは整数のリストなので、そのままテンソルに変換
    targets_tensor = torch.tensor(targets)
    
    return inputs_padded, targets_tensor



def random_drop_and_average(data_list, droprate):
    """
    ランダムにドロップして平均を取る関数
    これでstdを作る
    """
    if not 0 <= droprate <= 1:
        raise ValueError("droprate must be between 0 and 1")

    # Calculate the number of elements to drop
    num_to_drop = int(len(data_list) * droprate)
    
    # Randomly select indices to drop
    indices_to_drop = set(random.sample(range(len(data_list)), num_to_drop))
    
    # Filter the list to remove the selected indices
    filtered_list = [value for index, value in enumerate(data_list) if index not in indices_to_drop]
    
    # Calculate the average of the remaining elements
    if not filtered_list:
        return 0  # or handle the empty list case as needed
    return sum(filtered_list) / len(filtered_list)


def random_drop_accuracy(labels:np.ndarray,droprate:float) -> float:
    """
    ランダムにドロップして平均を取る関数
    これでstdを作る
    :param labels: (predict,target)のリスト
    :param droprate: ドロップする割合
    :return<float>: ドロップしたときのaccuracy
    """

    all_indeces=np.arange(len(labels))
    drop_indeces=np.random.choice(all_indeces,size=int(len(all_indeces)*droprate),replace=False)
    filtered_labels=np.array([
        label for index,label in enumerate(labels) if index not in drop_indeces
    ])

    acc=np.mean((filtered_labels[:,0]==filtered_labels[:,1]).astype(float))
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="configのあるパス")
    parser.add_argument("--device",default=0,help="GPUの番号")
    parser.add_argument("--timescale",default=1,type=float,help="何倍に時間をスケールするか. timescale=2でtimewindowが1/2になる.")
    parser.add_argument("--saveto",required=True,help="結果を保存するディレクトリ")
    parser.add_argument("--modelname",default="model_best.pth",help="モデルのファイル名")
    parser.add_argument("--is_video", action='store_true')
    parser.add_argument("--testnum",type=int,default=5,help="stdを求めるために何回testするか")
    parser.add_argument("--test_droprate",type=float,default=0.3,help="testデータにランダム性を持たせるために, 1minibatchごとにdropするrate")
    args = parser.parse_args()

    timescale=args.timescale
    testnum=args.testnum
    test_droprate=args.test_droprate

    model_dir=Path(args.target).name
    result_dir=Path(__file__).parent/(args.saveto)/model_dir

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf:dict
    train_conf,model_conf=conf["train"],conf["model"]

    # minibatch=train_conf["batch"]
    minibatch=16
    sequence=train_conf["sequence"] #時系列のタイムシーケンス
    #<< configの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="dynamic-snn".casefold():
        if "res".casefold() in model_conf["cnn-type"]:
            model=DynamicResCSNN(model_conf)
        else:
            model=DynamicCSNN(model_conf)
        criterion=SF.ce_rate_loss()
    elif model_conf["type"]=="snn".casefold():
        if "res".casefold() in model_conf["cnn-type"]:
            model=ResCSNN(model_conf)
        else:
            model=CSNN(model_conf)
        criterion=SF.ce_rate_loss()
    elif model_conf["type"]=="lstm".casefold():
        model=ResNetLSTM(model_conf)
        criterion=torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    print(model)
    modelname=args.modelname if ".pth" in args.modelname else args.modelname+".pth"
    model.load_state_dict(torch.load(Path(args.target)/f"result/{modelname}",map_location=device))
    model.to(device)

    scale_predictor=ScalePredictor(datatype="gesture")

    encoder=torch.nn.Identity() #デフォルトは何もしない
    print(f"encoder: {encoder}")
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_window=int(train_conf["timewindow"]/timescale) if "timewindow" in train_conf.keys() else int(train_conf["time-window"]/timescale)
    if model_conf["in-size"]==128:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000), #denoiseって結構時間かかる??
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
            torch.from_numpy,
        ])
    else:
        transform=torchvision.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(model_conf["in-size"],model_conf["in-size"])),
            tonic.transforms.ToFrame(sensor_size=(model_conf["in-size"],model_conf["in-size"],2),time_window=time_window),
            torch.from_numpy,
        ])


    cache_path=ROOT/f"data/cache-data/gesture/window{time_window}-insize{model_conf['in-size']}/test"
    testset=tonic.datasets.DVSGesture(save_to=ROOT/"data/original-data",train=False,transform=transform)
    testset=tonic.DiskCachedDataset(testset,cache_path=cache_path)
    test_loader = DataLoader(testset,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=3,drop_last=False)
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # Validation step
    print_terminal(f"eval model: {model_conf['type']}@ time-scale: {timescale}"+"-"*500)
    model.eval()

    val_loss = []
    labels=[]
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):

            inputs, targets = inputs.to(device).to(torch.float), targets.to(device)

            if sequence>0 and inputs.shape[1]>sequence*timescale: #configでシーケンスが指定された場合はその長さに切り取る
                inputs=inputs[:,:int(sequence*timescale)]

            if model_conf["type"]=="dynamic-snn":
                a=scale_predictor.predict_scale_trajectory(inputs.permute((1,0,*[i+2 for i in range(inputs.ndim-2)]))) #encoderに通す前に予測
                print(f"a: {a}")
            #>> encoderに通す >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            inputs=encoder(inputs)
            inputs[inputs>0.0]=1.0
            #<< encoderに通す <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            inputs=inputs.permute((1,0,*[i+2 for i in range(inputs.ndim-2)])) #[timestep x batch x x_dim]


            if not model_conf["type"]=="dynamic-snn":
                outputs = model(inputs)
            else:

                # outputs=model(inputs)

                # print(a[:int(50)])
                outputs=model.dynamic_forward_v1(
                    inputs,a=timescale*torch.ones(inputs.shape[0])
                )

                # outputs=model.dynamic_forward(
                #     s=inputs,scale_predictor=scale_predictor
                # )

            targets=targets.to(torch.long)
            val_loss.append(criterion(outputs, targets).item())

            if "snn".casefold() in model_conf["type"].casefold():
                predict_label=torch.argmax(torch.sum(outputs,dim=0),dim=1).to("cpu").detach().flatten().numpy()
            else:
                predict_label=torch.argmax(outputs,dim=1).to("cpu").detach().flatten().numpy()
            targets=targets.to("cpu").detach().flatten().numpy()
            print(f"predict_label: {predict_label}")
            print(f"targets: {targets}")
            acc=np.mean((predict_label==targets).astype(float))
            print(f"acc: {acc}")

            labels+=[
                (label_p,label_t) for label_p,label_t in zip(predict_label,targets)
            ]

    labels=np.array(labels)
    # print(f"labels: {labels}")
    # print(f"labels shape: {np.array(labels).shape}")

    acc_epoches=[]
    loss_epoches=[]
    for _ in range(testnum):
        acc_epoches.append(random_drop_accuracy(labels,test_droprate))
        loss_epoches.append(random_drop_and_average(val_loss,test_droprate))
    acc_mean=np.mean(acc_epoches)
    acc_std=np.std(acc_epoches)
    loss_mean=np.mean(loss_epoches)
    loss_std=np.std(loss_epoches)
    
    # print(f"acc_epoches: {acc_epoches}")
    # print(f"loss_epoches: {loss_epoches}")

    print_terminal(f"{testnum} Trial Result: ACC {acc_mean:.4f}±{acc_std:.4f}")
    print_terminal(f"done\n")


    ##>> 入力確認 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if args.is_video:
        print_terminal(f"saveing sample videos...")
        video_size=320
        sample_num=5
        for i_frame in tqdm(range(sample_num)):
            frame_np=inputs[:,i_frame].to("cpu").detach().numpy()
            frame=1.5*frame_np[:,0]+0.5*frame_np[:,1]-1
            save_heatmap_video(
                frame,
                output_path=result_dir/f"video",
                file_name=f"train_input_label{targets[i_frame]}",
                fps=60,scale=int(video_size/model_conf["in-size"])
            )
        print_terminal("done")
    ##<< 入力確認 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    result={
        "model":model_conf["type"],
        "model_dir":model_dir,
        "datatype":train_conf["datatype"],
        "time-scale":args.timescale,
        "acc_mean":acc_mean,
        "acc_std":acc_std,
        "loss_mean":loss_mean,
        "loss_std":loss_std,
    }
    save_dict2json(
        result,saveto=result_dir/f"result_ts{args.timescale:.2f}.json"
    )


if __name__=="__main__":
    main()