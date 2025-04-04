# Exp2: Gesture Recognition
`exp2` is codes for the exp2 in the paper.
~~~
exp2/
├── train/
├── eval/
│   ├── acc_eval/
│   └── mem_eval/
├── readme.md
~~~

## train
Training codes are in the `train` directory.
```bash
python train.py --target {path to config PARENT path} --device 0 
# python train.py --target dynamic-snn/tau0.008 --device 0 
```
`configpath` is the path to the train config file.
There are some examples in the `train` directory.


## eval
Evaluation codes are in the `eval` directory.

### acc_eval
Accuracy evaluation codes are in the `acc_eval` directory.
```bash
python acc_eval/eval.py --target {path to model PARENT path} --device 0 --saveto dynamic-snn --is_video
# python acc_eval/eval.py --target /codes/experiments/exp2/train/dynamic_snn/tau0.008 --device 0 --saveto dynamic-snn --is_video
```


### mem_eval
Memory evaluation codes are in the `mem_eval` directory.
This code visualize the membrane potential of the model.
```bash
python mem_eval/eval.py --target {path to model PARENT path} --device 0 --saveto dynamic-snn
# python mem_eval/eval.py --target /codes/experiments/exp2/train/dynamic_snn/tau0.008 --device 0 --saveto dynamic-snn
```
