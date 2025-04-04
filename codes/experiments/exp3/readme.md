# Exp3: Trajectory Prediction
`exp2` is codes for the exp2 in the paper.

~~~
exp3/
├── env/
├── collect_traindata/
├── train/
├── eval/
│   ├── eval.py
│   └── visualize_trj.py
├── readme.md
├── exp3_utils.py
├── trajectory_generator.py
~~~

## env
Experiment environment is in the `env` directory.   
No need to change and excute the code in `env` directory.

## collect_traindata
You can collect train data by running the code in the `collect_traindata` directory.
~~~bash
python collect_traindata/run.py --configpath {path to config file}
# python collect_traindata/run.py --configpath collect_traindata/config.yml
~~~
The collected data is saved in `data/original-data/Trajectory` directory.

## train
Training codes are in the `train` directory.
```bash
python train/train.py --target {path to config PARENT path} --device 0 
# python train/train.py --target train/dynamic-snn/tau0.008 --device 0 
```
There are some example config files in the `train` directory.

## eval
Evaluation codes are in the `eval` directory.
### eval.py
```bash
python eval/eval.py --modelpath {path to model PARENT path} --trjpath {path to trajectory csv data path} --saveto {path to save result}
# python eval/eval.py --modelpath /codes/experiments/exp3/train/dynamic-snn/tau0.06 --trjpath /data/original-data/Trajectory/datasets.csv --saveto dynamic-snn/tau0.06
```

### visualize_trj.py
Visualize the trajectory by running `eval.py` as mp4 file.
```bash
python eval/visualize_trj.py --trjpath {path to trajectory csv PARENT path} 
# python eval/visualize_trj.py --trjpath dynamic-snn/tau0.06/trajectory_reslut
```
