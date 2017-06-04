# Adversarial Transformation Network(ATN)

## Introduction

  A simple implement of an Adversarial Autoencoding ATN(AAE ATN) proposed in [Adversarial Transformation Networks: Learning to Generate Adversarial Examples](https://arxiv.org/abs/1703.09387) using tensorflow.

## Requirements

python 3.5
tensorflow 1.1.0
matplotlib (for result visualizing)

## Usage

  You can test with my trained model:

  ```shell
  python atn.py
  ```

  If you want to train by yourself:
  ```shell
  python atn.py --train
  ```

## Result

  Here are some visualized samples:

  ![result](./result.png)

  Before attack, the accuracy of the target cnn network is **0.9902**, and it becomes **0.4756** after attack.

  The result is not good enough, so **WELCOME CONTRIBUTION !!!**
