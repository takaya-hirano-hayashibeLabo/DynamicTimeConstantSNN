# Enhancing Time Scalability of Spiking Neural Networks with Dynamic Time Constants
This repository contains the code for the paper "Enhancing Time Scalability of Spiking Neural Networks with Dynamic Time Constants".
The trained models and training data are too large to be uploaded to the repository.
You need to download and train the models by yourself.
(When you execute the code, it will automatically download the data from the internet or collect the data using a simulator.)

## Installation
~~~bash
pip install -r requirements.txt
~~~
Please install `torch` that is compatible with your cuda version if you want to use GPU.

## Directory Stracture
~~~
├── codes/
│   ├── experiments/
│   │   ├── exp1/
│   │   ├── exp2/
│   │   └── exp3/
│   ├── models/
│   └── utils.py
├── data/
├── requirements.txt
└── readme.md
~~~
### data/
Initially, `data/` directory is empty.
Training data is saved in `data/` when you run the code.

### codes/
`codes/` directory contains the code for the paper.

#### codes/models/
`models/` directory contains the code for the SNN models and so on.

#### codes/experiments/
`experiments/` directory contains the code for the experiments.
Numbers of experiments are compatible with the paper.
Usage of the code in `experiments/` is in each `exp` directory as `readme.md`.






