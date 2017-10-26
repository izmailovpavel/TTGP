# Tensor Train Gaussian Processes in TensorFlow.

This repository contains the Tensorflow implementation of the TTGP method (see [Scalable Gaussian Processes with Billions of Inducing Inputs via Tensor Train Decomposition](https://arxiv.org/abs/1710.07324) paper) by Pavel Izmailov, Alexander Novikov and Dmitry Kropotov.

## Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
4. [Usage](#usage)
    1. [Installation](#installation)
    2. [Synthetic Data](#synthetic-data)
    3. [Examples: MNIST, CIFAR10, CelebA, SVHN](#mnist-cifar10-celeba-svhn)
    4. [Custom data](#custom-data)
    
## Introduction

TT-GP is a scalable GP method based on inducing inputs, stochastical variational inference and the Tensor Train decomposition.
Unlike other existing methods, TT-GP is scalable both in terms of the number of data points and in terms of the number of inducing inputs. Using a lot of inducing inputs allows to capture complex structure in data and learn expressive kernel functions (including deep kernels), which is particularly important for large datasets.

In this repo we provide the implementation of TT-GP with RBF and Deep kernels. We also attach scripts for running all the experiments from the paper.

## Dependencies

...

## Usage

...
