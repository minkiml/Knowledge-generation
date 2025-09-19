# Mode Connectivity
This folder contains codes for studying "neural network mode connectivity" and running some tests.

## What is mode connectivitiy? 
In the intricate world of deep learning, neural network mode connectivity sheds light on a fascinating property of the training process: the ability to find "continuous paths of low error" between different successful solutions, or "modes," that a neural network discovers.

Especially, the groundbreaking discovery of mode connectivity revealed that the isolated valleys (simply a local or global minimum in loss landscape) are often not actually isolated. Instead, they are frequently connected by "mountain passes" or "ridges" where the loss remains surprisingly low. In other words, it's possible to walk from one good solution to another without a significant increase in error (even without training again), potentially implying that the different neural networks might be all connected in very high-dimensional parameter space. This concept is crucial for understanding the geometry of the high-dimensional loss landscapes that these models navigate and has significant practical implications for how the knowledge (neural networks) can be made, transformed, elaborated, and controlled.

