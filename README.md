# CG-like-Adam
Code of CG-like-Adam optimization algorithm for training deep networks.

The paper "Conjugate-Gradient-like Based Adaptive Moment Estimation Optimization Algorithm for Deep Learning" of this CG-like-Adam optimization algorithm is under submitting.

Training deep neural networks is a challenging task. In order to speed up training and enhance the performance of deep neural networks, we rectify the vanilla conjugate gradient as conjugate-gradient-like and incorporate it into the generic Adam, and thus propose a new optimization algorithm named CG-like-Adam for deep learning. Specifically, both the first-order and the second-order moment estimation of generic Adam are replaced by the conjugate-gradient-like. Convergence analysis handles the cases where the exponential moving average coefficient of the first-order moment estimation is constant and the first-order moment estimation is unbiased. Numerical experiments show the superiority of the proposed algorithm based on the CIFAR10/100 dataset.
