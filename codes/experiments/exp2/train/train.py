import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))

import os
import torch
import numpy as np
from snntorch import functional as SF
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import torchvision
import tonic
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from math import floor
from datetime import datetime


from utils import load_yaml,print_terminal,calculate_accuracy,save_dict2json
from models import DynamicCSNN,CSNN,DynamicResCSNN, ResCSNN, ResNetLSTM


DATAPATH=ROOT.parent/"data"


def plot_and_save_curves(result, resultpath, epoch):
    df = pd.DataFrame(result, columns=["epoch", "datetime","train_loss_mean", "train_loss_std", "train_acc_mean", "train_acc_std", "val_loss_mean", "val_loss_std", "val_acc_mean", "val_acc_std"])
    
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="configのあるパス")
    parser.add_argument("--device",default=0,help="GPUの番号")
    args = parser.parse_args()


    #>> configの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device=torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf,model_conf=conf["train"],conf["model"]

    epoch=train_conf["epoch"]
    iter_max=train_conf["iter"]
    save_interval=train_conf["save_interval"]
    minibatch=train_conf["batch"]
    base_timewindow=train_conf["timewindow"]
    base_sequence=train_conf["sequence"]
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
    model.to(device)

    weight_decay=train_conf["weight-decay"] if "weight-decay" in train_conf else 0.0
    model_params=model.split_weight_decay_params( #正則化をかけるパラメータとかけないパラメータを分ける
        no_decay_param_names=["w","bias"], #wは時定数の逆数
        weight_decay=weight_decay
    )

    optim_type=train_conf["optim"] if "optim" in train_conf else "Adam"
    if optim_type=="Adam":
        optim=torch.optim.Adam(model_params,lr=train_conf["lr"])
    elif optim_type=="AdamW": #weight decayを利用する場合は, weight decayを正しく実装したAdamWを使う
        optim=torch.optim.AdamW(model_params,lr=train_conf["lr"])


    if train_conf["schedule-step"]>0:   
        scheduler=StepLR(optimizer=optim,step_size=train_conf["schedule-step"],gamma=train_conf["schedule-rate"])
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    train_loaders=[]
    test_loaders=[]
    for timescale in train_conf["timescales"]:
        time_window=round(base_timewindow/timescale)
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


        trainset=tonic.datasets.DVSGesture(save_to=DATAPATH/"original-data",train=True,transform=transform)
        testset=tonic.datasets.DVSGesture(save_to=DATAPATH/"original-data",train=False,transform=transform)

        cachepath=DATAPATH/f"cache-data/{train_conf['datatype']}/window{time_window}-insize{model_conf['in-size']}"
        trainset=tonic.DiskCachedDataset(trainset,cache_path=str(cachepath/"train"))
        testset=tonic.DiskCachedDataset(testset,cache_path=str(cachepath/"test"))

        train_loaders.append({   
            "timescale":timescale,
            "loader":DataLoader(trainset, batch_size=minibatch, shuffle=True,collate_fn=custom_collate_fn ,num_workers=1)
        })
        test_loaders.append({
            "timescale":timescale,
            "loader":DataLoader(testset,   batch_size=minibatch, shuffle=False,collate_fn=custom_collate_fn,num_workers=1),
        })
    iter_n_train=len(train_loaders[0]["loader"]) #イテレーションの最大数
    iter_n_test =len( test_loaders[0]["loader"])
    #<< データの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> 学習ループ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    result=[]
    best_score={"mean":0.0, "std":0.0}
    for e in range(epoch):

        train_iters=[{"timescale":items["timescale"], "iter":iter(items["loader"])} for items in train_loaders]

        model.train()
        it=0
        train_loss_list=[]
        train_acc_list=[]
        for batch_idx in range(iter_n_train):

            # print(f"iter:{batch_idx}"+"-"*50)
            iter_loss=0
            iter_acc=0
            for items in train_iters:
                ts,train_iter=items["timescale"], items["iter"]
                inputs,targets=next(train_iter)
                inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device)
                inputs[inputs>0]=1.0

                sequence=round(base_sequence*ts) #timescaleに合わせてsequenceを調整
                if sequence>0 and inputs.shape[0]>sequence: #configでシーケンスが指定された場合はその長さに切り取る
                    inputs=inputs[:sequence]

                # print(f"timescale:{ts}, in shape:{inputs.shape}")
                # print(f"target exp:{targets[:10]}")

                outputs = model.forward(inputs)
                targets=targets.to(torch.long)
                loss:torch.Tensor = criterion(outputs, targets)
                loss.backward()
                model.clip_gradients() #勾配クリッピング
                optim.step()
                optim.zero_grad()
                iter_loss+=loss.item()

                if "snn".casefold() in model_conf["type"]:
                    iter_acc+=SF.accuracy_rate(outputs,targets)
                else:
                    iter_acc+=calculate_accuracy(outputs,targets)

            train_loss_list.append(iter_loss/len(train_iters))
            train_acc_list.append(iter_acc/len(train_iters))
            print(f"Epoch [{e+1}/{epoch}], Step [{batch_idx}/{iter_n_train}], Loss: {loss.item():.4f}")
            it+=1
            if iter_max>0 and it>iter_max:
                break
            
        if train_conf["schedule-step"]>0:   
            scheduler.step() #学習率の更新

        # Validation step
        model.eval()
        test_iters=[{"timescale":items["timescale"], "iter":iter(items["loader"])} for items in test_loaders]
        with torch.no_grad():
            val_loss = []
            test_acc_list=[]

            for it_test in range(iter_n_test):
                iter_loss=0
                iter_acc=0
                for items in test_iters:
                    ts,test_iter=items["timescale"],items["iter"]
                    inputs,targets=next(test_iter)
                    inputs, targets = inputs.to(device).permute((1,0,*[i+2 for i in range(inputs.ndim-2)])).to(torch.float), targets.to(device)
                    inputs[inputs>0]=1.0

                    sequence=round(base_sequence*ts) #timescaleに合わせてsequenceを調整
                    if sequence>0 and inputs.shape[0]>sequence: #configでシーケンスが指定された場合はその長さに切り取る
                        inputs=inputs[:sequence]
                    
                    outputs = model(inputs)
                    targets=targets.to(torch.long)
                    iter_loss+=criterion(outputs, targets).item()
                    if "snn".casefold() in model_conf["type"]:
                        iter_acc+=SF.accuracy_rate(outputs,targets)
                    else:
                        iter_acc+=calculate_accuracy(outputs,targets)

                val_loss.append(iter_loss/len(test_iters))
                test_acc_list.append(iter_acc/len(test_iters))

                if it_test>iter_max and iter_max>0:
                    break

            acc_mean,acc_std=np.mean(test_acc_list),np.std(test_acc_list)
            if acc_mean>best_score["mean"]: #テスト最高スコアのモデルを保存
                best_score["mean"]=acc_mean
                best_score["std"]=acc_std
                best_score["epoch"]=e
                save_dict2json(best_score,resultpath/f"best-score.json")
                torch.save(model.state_dict(),resultpath/f"model_best.pth")

            print(f"Validation Loss after Epoch [{e+1}/{epoch}]: {np.mean(val_loss):.4f}")

        # Save model checkpoint
        if (e + 1) % save_interval == 0:
            torch.save(model.state_dict(), resultpath / f"model_epoch_{e+1}.pth")

        result.append([
            e,
            datetime.now(),
            np.mean(train_loss_list), np.std(train_loss_list),
            np.mean(train_acc_list), np.std(train_acc_list),
            np.mean(val_loss),np.std(val_loss),
            np.mean(test_acc_list), np.std(test_acc_list)
        ])
        result_db = pd.DataFrame(
            result, 
            columns=["epoch","datetime","train_loss_mean", "train_loss_std", "train_acc_mean", "train_acc_std", "val_loss_mean","val_loss_std", "val_acc_mean", "val_acc_std"]
        )
        result_db.to_csv(resultpath / "training_results.csv", index=False)
        # Plot and save curves
        plot_and_save_curves(result, resultpath, e + 1)
    torch.save(model.state_dict(), resultpath / f"model_final.pth")
    #<< 学習ループ <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   


if __name__=="__main__":
    main()