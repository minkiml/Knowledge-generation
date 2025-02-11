# Merges

This folder contains full codes necessary for merging operation as well as running test.

## Components

- **merge_test.py**: All-in-one test script, where two LeNet models are independently trained on CIFAR-10, get merged, and provides a log folder.   
- **merge.py**: Contains the main merge class and run function.
- **utils.py**: Contains all the util functions used for merging (merge.py). 

- **matching.py**: Contain core codes for alignment.
    - **matching_functions.py**: Contains some of SOTA matching algorithms.
    - **metrics.py**: Contains a class used to compute similiarity metric from intermediate features. 
    - **rebasin.py**: Contain a class that handles all the matrix compositional operations for different nn modules. 

- **profiles.py**: Contain core codes for extracting network graph for merging. What to merge and not is determined here. 
    - **profile.py**: Extract profile (graph) from trained models necessary for merging.
    - **lookup.py**: Contain lists of NN classes that require extra care when merging. 
    
- **trainings**
    - **trainings.py**: Training function. 
    - **validation.py**: Validation function.

- **training_utils**: Contain all the util functions and classes used for training and logging.

- **dataset_merge**: Contain vision detaset loaders to test merging.

## How to run test

1. pip install -r merges/requirements.txt
2. Specify a directory (--data_path) in [here](/merges/merge_test.py) for downloading in and loading datasets from. 
3. in the main root directory, run python -m merges.merge_test 

## TODOs

- Nested merging and merging more than 2 at the same time.
- Implement child NNprofile for Transformers and LoRA modules. 
- SVD-based alignment for LoRA (**Model Merging with SVD to Tie the Knots**).
- Sanity check CCA and Zipit alignment algorithms.
- Weight matching algorithms for alignment.
- Visualizations - 1d & 2d.
- Matching function (TIES, Arithmetics, etc.).