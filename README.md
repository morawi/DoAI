# DoAI

Do AI models learn better if they see more classes?

(SWENG 2021 Group 6)

## Overview

We use classification accuracy metric in machine learning to measure model performance.
However, this metric can be misleading for many reasons: data imbalance, percentage of samples incorrectly classified and number of classes.
It would be interesting to address the latter.

Now, the simplest (computer vision) classification problems to consider are the `CIFAR-10` and `CIFAR-100` datasets.
The best reported performance is through this work and the code is available in PyTorch (the proposed model achieved 99.70% and 96.08 in CIFAR-10 and CIFAR-100, respectively).

## The research question 

Does the model do a better job on CIFAR-10 (has 10 classes) or CIFAR-100 (has 100 classes)?

To answer this question, we need to know how to statistically quantify the Model Performance when the number of classes/categories changes. 

## Methods

This work is straightforward: data and code are publicly available. PyTorch platform supports data reading and deep learning models. We use the t-test statistic to investigate the problem.

We built the tests on top of PyTorch models. It would be interesting to test other data sets containing more categories.
