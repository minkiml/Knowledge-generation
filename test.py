import os
import argparse
import torch
import numpy as np
import logging
from copy import deepcopy
from tqdm import tqdm

from Hypernetwork.hn_framework import Framework_HN
import merging as mg
from NNConnectivitiy.NNC_analysis import plot_1d
if __name__ == "__main__":
        # Train hypernet
    
    network = mg.LeNet(input_dim = 3, output_dim = 10, dataset="CIFAR-10", batchnorm = True)
    hypernet = Framework_HN(network)
    
    # print(hypernet.implicitnet)
    
    loss = hypernet.forward_Hypernet("updating")
    print(loss)
    print(hypernet.implicitnet)
    # print("")
    # print(hypernet.hypernet.projections)
    # print("")
    # print(hypernet.hypernet.depth)