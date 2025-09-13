import os
import torch
import numpy as np
import copy
from tqdm import tqdm
import merging as mg
import torch.nn.functional as F


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.rcParams['agg.path.chunksize'] = 1000
sns.set(style='ticks', font_scale=1.2)
plt.rcParams['figure.figsize'] = 12,8

def eval_grad_training(hypernet, sampler, model_frame, testing_data, device, logger, path):
    logger.info("Evaluating Grad-trained hypernet ")
    
    
    # 1. Sample from 0 to 1 gradually 
    last = 2.0
    alpha_list = torch.linspace(0, last, steps=40)
    acc_list = []
    
    for a in tqdm(alpha_list, desc = "Eval ..."):
        t = sampler.sample(a)
        
        network = hypernet.marterialize_Implicitnet(copy.deepcopy(model_frame), t = t)
        acc = mg.vali(network.to(device), mg.merge_data_iter(testing_data), device)
        acc_list.append(acc)
    
    
    alpha_arr = alpha_list.numpy()
    acc_arr = np.array(acc_list)
    mask_red = alpha_arr <= 1.0
    mask_blue = alpha_arr > 1.0
    
    split_idx = np.argmax(mask_blue)
    # Construct red segment (up to and including split point)
    alpha_red = alpha_arr[:split_idx + 1]
    acc_red = acc_arr[:split_idx + 1]

    # Construct blue segment (starting from split point)
    alpha_blue = alpha_arr[split_idx:]
    acc_blue = acc_arr[split_idx:]

    fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
    axes = fig.subplots()
    axes.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # Plot with dashed lines and markers
    # axes.plot(alpha_list, acc_list, 'o-', label="Accuracy", alpha = 0.8, color = "red")    
    axes.plot(alpha_red, acc_red, 'o-', color='red', label='ID')
    axes.plot(alpha_blue, acc_blue, 'o-', color='blue', label='OOD')

    plt.xlabel('Trajectory (t)', fontsize=20)
    plt.ylabel('ACC (%)', fontsize=20)
    # Set x-ticks and labels
    xticks_positions = np.linspace(0,last, int(a / 0.2) + 1) # inverval is 0.1 #np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(ticks=xticks_positions, fontsize=20, fontweight='bold')
    # xticks_labels = [r'$\theta_{g \rightarrow s}$', '0.2', '0.4', '0.6', '0.8', r'$\theta_{g \rightarrow p}$']
    
    # Set y-ticks from 0 to 100 with intervals of 25
    yticks_positions = np.arange(0, 101, 25)
    plt.yticks(ticks=yticks_positions, fontsize=20, fontweight='bold')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5)
    # Find the indices of the best values in acc_p_list and acc_s_list
    best_p_idx = np.argmax(acc_list)
    # Get corresponding alpha values for best points
    best_p_alpha = alpha_list[best_p_idx]
    legend = axes.legend(fontsize=20, loc='upper left', bbox_to_anchor=(0.5, 1.1))
    # Annotate the best values
    axes.annotate(f'Best: {acc_list[best_p_idx]:.2f}%',
                xy=(best_p_alpha, acc_list[best_p_idx]),
                xytext=(best_p_alpha, acc_list[best_p_idx] + 2),
                textcoords='data', ha='center', fontsize=14, color='red',
                arrowprops=dict(facecolor='red', arrowstyle="->"))
    
    plt.savefig(os.path.join(path, "grad_eval.png" ), bbox_inches='tight') 
    plt.clf()   
    plt.close(fig) 
    
    
#######################################################################################################################################
#######################################################################################################################################


def eval_grad_learning(hypernet, sampler, model_frame, testing_data, device, logger, path,
                       predefined_ipe):
    logger.info("Evaluating Grad-learned hypernet ")
    
    
    # 1. Sample from 0 to 1 gradually 
    last = 2.0
    alpha_list = []
    acc_list = []
    next_params = None
    for i in tqdm(range(predefined_ipe+1 + int(predefined_ipe*0.1)), desc="data free upadating"):
        a = (i) / predefined_ipe

        t = sampler.sample(a)
        
        network, next_params = hypernet.marterialize_Implicitnet_grad_learning(copy.deepcopy(model_frame), t = t, with_params = next_params)
        acc = mg.vali(network.to(device), mg.merge_data_iter(testing_data), device)
        acc_list.append(acc)
        alpha_list.append(a)
    
    
    
    alpha_arr = np.array(alpha_list)
    acc_arr = np.array(acc_list)
    mask_red = alpha_arr <= 1.0
    mask_blue = alpha_arr > 1.0
    
    split_idx = np.argmax(mask_blue)
    # Construct red segment (up to and including split point)
    alpha_red = alpha_arr[:split_idx + 1]
    acc_red = acc_arr[:split_idx + 1]

    # Construct blue segment (starting from split point)
    alpha_blue = alpha_arr[split_idx:]
    acc_blue = acc_arr[split_idx:]

    fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
    axes = fig.subplots()
    axes.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # Plot with dashed lines and markers
    # axes.plot(alpha_list, acc_list, 'o-', label="Accuracy", alpha = 0.8, color = "red")    
    axes.plot(alpha_red, acc_red, 'o-', color='red', label='ID')
    axes.plot(alpha_blue, acc_blue, 'o-', color='blue', label='OOD')

    plt.xlabel('Trajectory (t)', fontsize=20)
    plt.ylabel('ACC (%)', fontsize=20)
    # Set x-ticks and labels
    xticks_positions = np.linspace(0,last, int(a / 0.2) + 1) # inverval is 0.1 #np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(ticks=xticks_positions, fontsize=20, fontweight='bold')
    # xticks_labels = [r'$\theta_{g \rightarrow s}$', '0.2', '0.4', '0.6', '0.8', r'$\theta_{g \rightarrow p}$']
    
    # Set y-ticks from 0 to 100 with intervals of 25
    yticks_positions = np.arange(0, 101, 25)
    plt.yticks(ticks=yticks_positions, fontsize=20, fontweight='bold')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5)
    # Find the indices of the best values in acc_p_list and acc_s_list
    best_p_idx = np.argmax(acc_list)
    # Get corresponding alpha values for best points
    best_p_alpha = alpha_list[best_p_idx]
    legend = axes.legend(fontsize=20, loc='upper left', bbox_to_anchor=(0.5, 1.1))
    # Annotate the best values
    axes.annotate(f'Best: {acc_list[best_p_idx]:.2f}%',
                xy=(best_p_alpha, acc_list[best_p_idx]),
                xytext=(best_p_alpha, acc_list[best_p_idx] + 2),
                textcoords='data', ha='center', fontsize=14, color='red',
                arrowprops=dict(facecolor='red', arrowstyle="->"))
    
    plt.savefig(os.path.join(path, "grad_eval.png" ), bbox_inches='tight') 
    plt.clf()   
    plt.close(fig) 