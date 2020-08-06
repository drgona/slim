# SLiM: Structured Linear Maps

Drop in replacements for pytorch nn.Linear for stable learning and inductive priors 
in physics informed machine learning applications.

## install dependencies via .yml
$ conda env create -f env.yml

## Install dependencies manually

$ conda create -n slim python=3.7
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
