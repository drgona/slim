# SLiM: Structured Linear Maps

Drop in replacements for pytorch nn.Linear for stable learning and inductive priors 
in physics informed machine learning applications.

## [Complete documentation](https://pnnl.github.io/slim/)

## install dependencies via .yml
```console
$ conda env create -f env.yml
```

## Install dependencies manually

```console
$ conda create -n slim python=3.7
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
$ pip install python-mnist
```

## For developers

Continuous documentation integration was set up using the tutorial found here: 
https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1/

## Benchmarks

To run the MNIST benchmarks download [MNIST](http://yann.lecun.com/exdb/mnist/) 