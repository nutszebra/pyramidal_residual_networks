# What's this
Implementation of Pyramidal Residual Networks by chainer  

# Dependencies

    git clone https://github.com/nutszebra/pyramidal_residual_networks.git
    cd pyramidal_residual_networks
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation.  
* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  


# Cifar10 result
| network                                   | alpha  | depth  | total accuracy (%) |
|:------------------------------------------|--------|--------|-------------------:|
| Pyramidal Residual Networks [[1]][Paper]  | 270    | 110    | 96.23              |
| my implementation                         | 270    | 110    | 95.9               |

<img src="https://github.com/nutszebra/pyramidal_residual_networks/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/pyramidal_residual_networks/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
Deep Pyramidal Residual Networks [[1]][Paper]

[paper]: https://arxiv.org/abs/1610.02915 "Paper"
