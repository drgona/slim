# SLiM: Structured Linear Maps

Drop in replacements for pytorch nn.Linear for stable learning and inductive priors 
in physics informed machine learning applications.

## Install

$ conda create -n slim python=3.7
$ conda install pytorch # see https://pytorch.org/ for system specific install instructions
$ conda install scipy
$ conda install numpy
$ cd butterfly/factor_multiply
$ python setup.py develop