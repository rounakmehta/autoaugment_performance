# Practical AutoAugment: Measuring the Impact of AutoAugment on Classification Performance for Real World Image Datasets

### Final Results 

In this paper, we put ourselves in the shoes of practioners seeking to use AutoAugment to improve image classification performance on their own real-world datasets. That exploration included measuring the impact of using transfer learning on policies on more datasets, and showing it is possible to automatically find policies that improve classifier performance without the complexities of reinforcement learning used in AutoAugment.

The notebook with the final results can be found at https://github.com/rounakmehta/autoaugment_performance/blob/master/paper/Final%20Results.ipynb

The paper can be found at 
https://github.com/rounakmehta/autoaugment_performance/blob/master/paper/final.pdf

### Intro

AutoAugment authors show how automatically selected policies for data augmentation can improve on benchmark classifier performance on CIFAR-10, CIFAR- 100, SVHN, and ImageNet datasets. This repo contains code that can be used to reproduce results presented by Team 18 as part of the final project for COMS 4995.  

We aim to understand the impact of a AutoAugment approach on additional image datasets through :

* “transfer learning” of optimal policies for found by AutoAugment for CIFAR- 10 and SVHN and,
* optimal policy search on these additional image datasets using a “Simplified” AutoAugment approach based on Random Search.

The additional datasets considered are the “Iceberg” dataset from the Kaggle “Statoil/C-CORE Iceberg Classifier” Challenge and the “Quickdraw” dataset from the Kaggle “Quick, Draw! Doodle Recognition Challenge” Challenge. We also use CIFAR-10 and CIFAR-100 to benchmark the Simplified AutoAugment approach explored in this paper.


### Directory Structure 

The repo is organised as follows: 

* original_autoaugment_implementation : clone of the github repo with the original paper
* paper : contains reports and results for submissions 
* random_search : modified code to implement random search to find optimal policies
* rpmcruz-autoaugment: clone of rpmcruz/autoaugment repo
* transfer_learning/iceberg :  Kaggle datasets and baseline kernels for “Statoil/C-CORE Iceberg Classifier” Challenge
* transfer_learning/quickdraw : Kaggle datasets and baseline kernels for “Quick, Draw! Doodle Recognition Challenge” Challenge

### Prerequisites


1. Install TensorFlow

2. Download CIFAR-10/CIFAR-100 dataset


```
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
curl -o cifar-100-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
```

3. Download Kaggle Datasets to the respective directories. These instructions can be found in the data_readme.txt files in teach of the sub directories of the transfer_learning directory


## Authors

Quoc N. Le, Rounak Mehta, Vikram Natarajan, Anna Novakovska Columbia University {qnl2000,rm3652,vsn2113,an2915}@columbia.edu


## References

* E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le. Autoaugment: Learning augmentation policies from data, 2018.
* B. Zoph, V. Vasudevan, J. Shlens, Q. V. Le. Learning Transferable Architectures for Scalable Image Recognition, 2017.
* Ricardo Cruz, AutoAugment implementation, GitHub repository, https://github.com/rpmcruz/autoaugment, 2018
* H. Mania, A. Guy, and B. Recht. Simple random search provides a competitive approach to reinforcement learning, 2018.
* M. Kumar, G. E. Dahl, V. Vasudevan, and M. Norouzi. Parallel Architecture and Hyperparameter Search via Successive Halving and Classification, 2018.
* Vrany, Jirka. My best single model - simple CNN - LB 0.1541. https://www.kaggle.com/jirivrany/my-best-single-model- simple-cnn-lb-0-1541, 2018.
* A. Howard, et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, 2017.
* Beluga. Greyscale MobileNet [LB=0.892]. https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb- 0-892, 2018.
