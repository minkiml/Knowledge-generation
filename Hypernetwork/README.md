# Hypernetworks
This folder contains codes for creating hypernetwork. The codes can be easily called and used for other frameworks (e.g., subspace network merging and mode connectivity) available in this repository.

## What is Hypernetworks
Hypernetworks are a neural network that generates the weights (and sometimes biases) for another, typically larger (can also be much smaller) neural network called the main network. They are generally used for parameter compression or dynamic and adaptive model generation.  

## About the codes and running
The codes here are used to run and study two different experiments. To run the codes, first download datasets (e.g., MINIST) and specify the data directory to the run script in scripts folder.  

1. This is to study training a tartget main network by using a hypernetwork and constructing a gradient trajectory (see [main_hn_grad.py](main_hn_grad.py)).

```train-and-eval
sh ./Hypernetwork/scripts/hn_mnist_grad.sh
```

2. This is to study "merging" and mode connectivity in subspace and investigate how subspace merging and connectivity could be related to merging and connectivity in the original paramemter space (see [main_hn.py](main_hn.py)).

```train-and-eval
sh ./Hypernetwork/scripts/hn_mnist.sh
```

Note that one challenge with hypernetworks is how to design an expressive and efficent hypernetwork to generate parameters in a structured way. In this repo, we use a Transformer to design the hypernetwork, where the sequence dimension represents each instance of dense matrix of the target main network and the hidden dimension gets projected to represent the dense matrix (input and output dimensions) of each instance. Our hypernets are designed to generate the entire parameters of target networks.
