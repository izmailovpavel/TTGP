# Tensor Train Gaussian Processes in TensorFlow.

__Note:__ this tutorial is under construction. We will update the tutorial soon.

This repository contains the Tensorflow implementation of the TTGP method (see [Scalable Gaussian Processes with Billions of Inducing Inputs via Tensor Train Decomposition](https://arxiv.org/abs/1710.07324) paper) by Pavel Izmailov, Alexander Novikov and Dmitry Kropotov.

## Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Usage](#usage)
    1. [Installation](#installation)
    2. [Examples: Airline, MNIST](#experiments-from-the-paper-mnist-airline-etc)
    3. [Custom data](#custom-data)
    
## Introduction

TT-GP is a scalable GP method based on inducing inputs, stochastical variational inference and the Tensor Train decomposition.
Unlike other existing methods, TT-GP is scalable both in terms of the number of data points and in terms of the number of inducing inputs. Using a lot of inducing inputs allows to capture complex structure in data and learn expressive kernel functions (including deep kernels), which is particularly important for large datasets.

In this repo we provide the implementation of TT-GP with RBF and Deep kernels. We also attach scripts for running all the experiments from the paper.

## Dependencies
This code has the following dependencies (version number crucial):

- Python 3
- Tensorflow 1.3
- scikit-learn
- numpy
- t3f (see https://github.com/Bihaqo/t3f for installation instructions)

## Usage

### Installation

### Experiments from the paper: Mnist, Airline, etc.

For experiments that are present in the paper we attach scripts to reproduce them.  
The files related to experiments with dataset `DS` are located in `TTGP/TTGP_experiments/D`
folder. For example,
to run the experiment on MNIST you should first prepare the data by going through the 
ipython notebook `TTGP/TTGP_experiments/mnist_60K_784/mnist_data.ipynb`, and then run 
the script `TTGP/TTGP_experiments/mnist_60K_784/experiment_class_gpnn.py`:
```
python3 TTGP/TTGP_experiments/mnist_60K_784/experiment_class_gpnn.py
```

Some folders contain scripts for different experiments, e.g. 
`TTGP/TTGP_experiments/mnist_60K_784/experiment_class_nn.py` allows to run just the neural network without a GP.
For the Airline data `experiment_class_gpnn.py` allows to run TT-GP with an RBF kernel, while `experiment_class_gpnn.py`
runs the method with a deep kernel.
