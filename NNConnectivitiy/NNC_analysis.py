import torch
import os
import copy
import numpy as np
import merging as mg 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tqdm import tqdm
# Set plt params
plt.rcParams['agg.path.chunksize'] = 1000
sns.set(style='ticks', font_scale=1.2)
plt.rcParams['figure.figsize'] = 12,8

def Interpolation(net_A, net_B,
                alpha = 0.5):
    # TODO implement partial merging
    state_dict_A = net_A.state_dict()
    state_dict_B = net_B.state_dict()
    state_dict_C = copy.deepcopy(state_dict_A)
    for key in state_dict_A:
        state_dict_C[key] = ((1 - alpha) * state_dict_A[key]) + (alpha * state_dict_B[key])
    
    intp_model = copy.deepcopy(net_A)
    intp_model.load_state_dict(state_dict_C)
    return intp_model

def plot_1d(model_A, model_B,
            testing_data,
            num_alpha,
            log_plot_path,
            device,
            title = "",
            testing_data_B = None):
    model_A.eval() # 0
    model_B.eval() # 1
    
    # get interpolation alpha
    alpha_list = torch.linspace(0, 1, steps=num_alpha)
    
    # interpolation and get accuracy
    acc_list = []
    acc_list_B = []
    for alpha in tqdm(alpha_list, desc = "1d plot: "):
        model_int = Interpolation(model_A, model_B, alpha)
        acc = mg.vali(model_int, mg.merge_data_iter(testing_data), device)
        acc_list.append(acc)
        if testing_data is not None:
            acc_B = mg.vali(model_int, mg.merge_data_iter(testing_data_B), device)
            acc_list_B.append(acc_B)
        
    fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
    axes = fig.subplots()
    # Plot with dashed lines and markers
    axes.plot(alpha_list, acc_list, 'o-', label="Accuracy on A", color = "red", alpha = 0.6)    
    if acc_list_B is not None:
        axes.plot(alpha_list, acc_list_B, 'o-', label="Accuracy on B", color = "blue", alpha = 0.6)    
    plt.xlabel(r'$\alpha$', fontsize=20)
    plt.ylabel('ACC (%)', fontsize=20)
    # Set x-ticks and labels
    xticks_positions = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
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
    legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1))
    # Annotate the best values
    axes.annotate(f'Best: {acc_list[best_p_idx]:.2f}%',
                xy=(best_p_alpha, acc_list[best_p_idx]),
                xytext=(best_p_alpha, acc_list[best_p_idx] + 2),
                textcoords='data', ha='center', fontsize=14, color='red',
                arrowprops=dict(facecolor='red', arrowstyle="->"))
    
    plt.savefig(os.path.join(log_plot_path, f"1d_lmc_{title}.png" ), bbox_inches='tight') 
    plt.clf()   
    plt.close(fig) 