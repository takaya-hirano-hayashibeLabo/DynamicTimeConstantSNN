# Exp1: Verification of the Proposed Theory
`exp1` is codes for the exp1 in the paper.
~~~
exp1/
├── linear/
├── cnn/
├── readme.md
~~~

## linear
```bash
python run_linear.py --timescale 3 --tau 0.008
```

## cnn
You can choose the model type from `csnn`, `csnn_dropout`, `rescsnn`.
```bash
python run_cnn.py --modeltype csnn --timescale 3 --tau 0.008
```



