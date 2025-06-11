import torch 
import numpy as np
import random
from torch.utils.data import Subset

def help_label_restructuring(total_train, total_test,
                             
                             total_label_num = 10,
                             split_p = [None, None], fixed_label_set = False):
    def create_relabel_subset(dataset, label_subset, label_map):
        # Get all indices where the target (label) belongs to the label subset
        indices = [i for i, (_, label) in enumerate(dataset) if label in label_subset]
        # Relabel the subset
        subset_relabel = [(data, label_map[label]) for data, label in Subset(dataset, indices)]
        return subset_relabel
    
    if split_p[0] is not None and split_p[1] is not None:
        assert split_p[0] + split_p[1] == total_label_num
        
        num_subset_A = split_p[0]
        num_subset_B = split_p[1]

    else:
        raise ValueError("Please specify the number of subsets desired")
    
    all_labels = list(range(total_label_num))

    # Shuffle the list of labels randomly
    random.shuffle(all_labels)

    # Randomly split the labels into two groups
    label_subset_A = [1, 4, 0, 9, 3] if fixed_label_set else all_labels[:num_subset_A]
    label_subset_B = [5, 2, 7, 6, 8] if fixed_label_set else all_labels[num_subset_A:num_subset_A + num_subset_B]

    # print(f"Labels in A: {label_subset_A}")
    # print(f"Labels in B: {label_subset_B}")

    # Create a mapping from original labels to new labels
    label_map_A = {old_label: new_label for new_label, old_label in enumerate(label_subset_A)}
    label_map_B = {old_label: new_label for new_label, old_label in enumerate(label_subset_B)}

    ############################
    ############################


    # Create dataset A and B for both training and testing sets
    train_A = create_relabel_subset(total_train, label_subset_A, label_map_A)
    train_B = create_relabel_subset(total_train, label_subset_B, label_map_B)

    test_A = create_relabel_subset(total_test, label_subset_A, label_map_A)
    test_B = create_relabel_subset(total_test, label_subset_B, label_map_B)


    return [train_A, test_A, label_subset_A], [train_B, test_B, label_subset_B]