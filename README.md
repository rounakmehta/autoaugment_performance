# Measuring the Impact of AutoAugment on Classification Performance for Additional Image Datasets

### Milestone 1 Results 

The results for Milestone 1 are can be found at https://github.com/rounakmehta/autoaugment_performance/blob/master/paper/Milestone%201%20Results.ipynb

### Intro

AutoAugment authors show how automatically selected policies for data augmentation can improve on benchmark classifier performance on CIFAR-10, CIFAR- 100, SVHN, and ImageNet datasets. This repo contains code that can be used to reproduce results presented by Team 18 as part of the final project for COMS 4995.  

We aim to understand the impact of a AutoAugment approach on additional image datasets through :

* “transfer learning” of optimal policies for found by AutoAugment for CIFAR- 10 and SVHN and,
* optimal policy search on these additional image datasets using a “Simplified” AutoAugment approach based on Random Search.

The additional datasets considered are the “Iceberg” dataset from the Kaggle “Statoil/C-CORE Iceberg Classifier” Challenge and the “Human Protein” dataset from the Kaggle “Human Protein Atlas Image Classification” Challenge. We also use CIFAR-10 and CIFAR-100 to benchmark the Simplified AutoAugment approach explored in this paper.


### Directory Structure 

The repo is organised as follows: 

* original_autoaugment_implementation : clone of the github repo with the original paper
* paper : contains reports and results for submissions 
* random_search : modified code to implement random search to find optimal policies
* rpmcruz-autoaugment: clone of rpmcruz/autoaugment repo
* transfer_learning/iceberg :  Kaggle datasets and baseline kernels for “Statoil/C-CORE Iceberg Classifier” Challenge
* transfer_learning/protein : Kaggle datasets and baseline kernels for “Human Protein Atlas Image Classification” Challenge

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
