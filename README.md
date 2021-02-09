# DoAI

Do AI models learn better if they see more classes?

Overview
We use classification accuracy metric in Machine Learning to measure model performance. However, this metric can be misleading for many reasons: data imbalance, percentage of samples incorrectly classified and number of classes. It would be interesting to address the latter. Now, the simplest (computer vision) classification problems to consider are the  CIFAR-10 and CIFAR-100 datasets. The best reported performance is through this work  and the code is available in PyTorch (the proposed model achieved 99.70% and 96.08 in CIFAR-10 and CIFAR-100, respectively).



The research question 

Does the model do a better job on CIFAR-10 (has 10 classes) or CIFAR-100 (has 100 classes)?  To answer this question, we need to know how to statistically quantify the Model Performance when the number of classes/categories changes. 

 

Methods
This work is straightforward: data and code are publicly available. PyTorch platform supports data reading and deep learning models. Still, I can provide a novel technique that can be used to investigate the problem. Students need to write the code necessary to implement the technique, on top of PyTorch models, and run the experiments. Other data sets containing more categories will also be used.
