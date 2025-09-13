import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import copy
from tqdm import tqdm
import merging as mg
# Set plt params
plt.rcParams['agg.path.chunksize'] = 1000
sns.set(style='ticks', font_scale=1.2)
plt.rcParams['figure.figsize'] = 12,8

##############################################################################################################################
##############################################################################################################################

def weights_vis(m, m2, dir = "", title = ""):
    # print("!")
    # print(m)
    # print(type(m))         # <class '__main__.MyClass'>
    # print(type(m).__name__)
    # print(m2)
    # print(type(m2))         # <class '__main__.MyClass'>
    # print(type(m2).__name__)
    # for name, param in m.named_parameters():
    #     print(f"{name}:   {param.shape}")
    # for name, param in m2.named_parameters():
    #     print(f"{name}:   {param.shape}")
    for (name, param), (named_2, param_2) in tqdm(zip(m.named_parameters(), m2.named_parameters()), desc = "Weight Visualization"):
        print(f"name of params: {name}")
        param = param.clone().detach().requires_grad_(False)
        param_2 = param_2.clone().detach().requires_grad_(False)
        if param.dim() == 4:
            param = param.mean(-1).mean(-1) # (out, in)
            param_2 = param_2.mean(-1).mean(-1) # (out, in)
        elif param.dim() == 3:
            param = param.mean(-1) # (out, in)
            param_2 = param_2.mean(-1) # (out, in)
        elif param.dim() == 2:
            pass
        elif param.dim() == 1:
            param = param.unsqueeze(-1) # (out, 1)
            param_2 = param_2.unsqueeze(-1) # (out, 1)
        
        param = param.T.cpu().numpy()
        param_2 = param_2.T.cpu().numpy()

        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        im = axes.imshow(param, cmap='viridis', aspect='auto')
        # Set axis labels and title
        plt.ylabel("in_dim")
        plt.xlabel("out_dim")
        # Add a colorbar using the ScalarMappable
        cbar = plt.colorbar(im)
        # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
        plt.savefig(os.path.join(dir, f"{title}_m_weight_of_layer({name}).png")) 
        plt.clf()   
        plt.close(fig)
        
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        im = axes.imshow(param_2, cmap='viridis', aspect='auto')
        # Set axis labels and title
        plt.ylabel("channel")
        plt.xlabel("length")
        # Add a colorbar using the ScalarMappable
        cbar = plt.colorbar(im)
        # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
        plt.savefig(os.path.join(dir, f"{title}_m2_weight_of_layer({name}).png")) 
        plt.clf()   
        plt.close(fig)

        
        param = abs(param - param_2)
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        im = axes.imshow(param, cmap='viridis', aspect='auto')
        # Set axis labels and title
        plt.ylabel("in_dim")
        plt.xlabel("out_dim")
        # Add a colorbar using the ScalarMappable
        cbar = plt.colorbar(im)
        # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
        plt.savefig(os.path.join(dir, f"{title}_diff_weight_of_layer({name}).png")) 
        plt.clf()   
        plt.close(fig)

##############################################################################################################################
##############################################################################################################################
        
def HypernetInterpolation(net_A, net_B, inter_initparams = False,
                alpha = 0.5):
    # TODO implement partial merging
    # TODO don't we have to merge the subspace embeddings too?
    state_dict_A = net_A.state_dict()
    state_dict_B = net_B.state_dict()
    state_dict_C = copy.deepcopy(state_dict_A)
    for key in state_dict_A:
        state_dict_C[key] = ((1 - alpha) * state_dict_A[key]) + (alpha * state_dict_B[key])
    
    intp_model = copy.deepcopy(net_A)
    intp_model.load_state_dict(state_dict_C)
    
    # TODO in case they are not the same (e.g., learnable)
    subparam_A = net_A.get_subspace_emb(learnable = False) 
    subparam_B = net_B.get_subspace_emb(learnable = False)
    if not isinstance(subparam_A, dict):
        subparam_C = ((1 - alpha) * subparam_A) + (alpha * subparam_B)
    else: raise NotImplementedError("")
    intp_model.set_subspace_emb(subparam_C)
    
    if inter_initparams:
        init_param_A=copy.deepcopy(net_A.init_params)
        init_param_B=copy.deepcopy(net_B.init_params)
        init_param_C=copy.deepcopy(net_B.init_params)
        for name in (init_param_A):
            init_param_C[name] = ((1 - alpha) * init_param_A[name]) + (alpha * init_param_B[name])
        intp_model.reset_initparams(init_param_C)
    return intp_model

def Hypernet_plot_1d(hypernet_A, hypernet_B, model_frame,
            testing_data,
            num_alpha,
            log_plot_path,
            device,
            title = "",
            testing_data_B = None):
    hypernet_A.eval() # 0
    hypernet_B.eval() # 1
    
    # get interpolation alpha
    alpha_list = torch.linspace(0, 1, steps=num_alpha)
    
    # interpolation and get accuracy
    acc_list = []
    acc_list_B = []
    for alpha in tqdm(alpha_list, desc = "1d plot: "):
        model_int = HypernetInterpolation(hypernet_A, hypernet_B, alpha = alpha, inter_initparams = True)
        network = model_int.marterialize_Implicitnet(copy.deepcopy(model_frame))
        acc = mg.vali(network, mg.merge_data_iter(testing_data), device)
        acc_list.append(acc)
        if testing_data_B is not None:
            acc_B = mg.vali(network, mg.merge_data_iter(testing_data_B), device)
            acc_list_B.append(acc_B)
            
    fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
    axes = fig.subplots()
    # Plot with dashed lines and markers
    axes.plot(alpha_list, acc_list, 'o-', label="Accuracy on A", alpha = 0.6, color = "red")    
    if testing_data_B is not None:
        axes.plot(alpha_list, acc_list_B, 'o-', label="Accuracy on B", alpha = 0.6, color = "blue")    
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
    
    plt.savefig(os.path.join(log_plot_path, f"hypernet_1d_lmc_{title}.png" ), bbox_inches='tight') 
    plt.clf()   
    plt.close(fig) 
    
##############################################################################################################################
##############################################################################################################################

def SubspaceInterpolation(subparamA, subparamB,
                        alpha = 0.5):
    # TODO implement partial merging
    if not isinstance(subparamA, dict):
        subparamC = ((1 - alpha) * subparamA) + (alpha * subparamB)
    else:
        subparamC = copy.deepcopy(subparamA)
        
        for key in subparamA:
            subparamC[key] = ((1 - alpha) * subparamA[key]) + (alpha * subparamB[key])
    
    return subparamC

def Subspace_plot_1d(hypernet_A, hypernet_B, model_frame,
            testing_data,
            num_alpha,
            log_plot_path,
            device,
            title = "",
            testing_data_B = None):
    hypernet_A.eval() # 0
    hypernet_B.eval() # 1
    
    # get interpolation alpha
    alpha_list = torch.linspace(0, 1, steps=num_alpha)
    
    # get subspace params
    subparam_A = hypernet_A.get_subspace_emb(learnable = False) # could either a dictionary of "L" number of params tensors or a nn.parameter of shape (1,L,dz) 
    subparam_B = hypernet_B.get_subspace_emb(learnable = False)
    
    # interpolation and get accuracy 
    acc_list = []
    acc_list_B = []
    for alpha in tqdm(alpha_list, desc = "Subspace 1d plot: "):
        subparams_C = SubspaceInterpolation(subparam_A, subparam_B, alpha)
        
        # Generate a network from the sampled subparam C ("assume, in general setting, the hypernet A and B are identical")
        network = hypernet_A.marterialize_Implicitnet(copy.deepcopy(model_frame), emb_in = subparams_C)
        
        
        acc = mg.vali(network, mg.merge_data_iter(testing_data), device)
        acc_list.append(acc)
        if testing_data_B is not None:
            acc_B = mg.vali(network, mg.merge_data_iter(testing_data_B), device)
            acc_list_B.append(acc_B)
            
    fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
    axes = fig.subplots()
    # Plot with dashed lines and markers
    axes.plot(alpha_list, acc_list, 'o-', label="Accuracy on A", alpha = 0.6, color = "red")    
    if testing_data_B is not None:
        axes.plot(alpha_list, acc_list_B, 'o-', label="Accuracy on B", alpha = 0.6, color = "blue")    
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
    
    plt.savefig(os.path.join(log_plot_path, f"hypernet_1d_lmc_{title}.png" ), bbox_inches='tight') 
    plt.clf()   
    plt.close(fig) 
    
##############################################################################################################################
##############################################################################################################################
        
def HypernetInterpolation2(set_paramA, set_paramB, init_param, alpha = 0.5):

    set_paramC = copy.deepcopy(init_param)
    for i, (key) in enumerate(init_param):
        set_paramC[key] = ((1 - alpha) * set_paramA[i]) + (alpha * set_paramB[i]) + init_param[key]
    
    # for i, (wA, wB) in enumerate(zip(set_paramA, set_paramB, set_paramC)):
    #     set_paramC = ((1 - alpha) * wA) + (alpha * wB)
    
    return set_paramC


'''
interpolate only the output of hyper net while keeping the init params

this is more like generated net merging rather than hypernet merging

This can be only made when the initparams equal and not changing
'''
def Hypernet_plot_1d2(hypernet_A, hypernet_B, model_frame,
            testing_data,
            num_alpha,
            log_plot_path,
            device,
            title = "",
            testing_data_B = None):
    hypernet_A.eval() # 0
    hypernet_B.eval() # 1
    
    # get interpolation alpha
    alpha_list = torch.linspace(0, 1, steps=num_alpha)
    # get hyper out 
    hyperparam_A = hypernet_A.forward_Hypernet(mode = "hyperout")
    hyperparam_B = hypernet_B.forward_Hypernet(mode = "hyperout")
    init_param = copy.deepcopy(hypernet_A.init_params)
    
    # interpolation and get accuracy
    acc_list = []
    acc_list_B = []
    for alpha in tqdm(alpha_list, desc = "1d plot: "):
        model_int = HypernetInterpolation2(hyperparam_A, hyperparam_B, init_param, alpha)
        network = hypernet_A.marterialize_Implicitnet(copy.deepcopy(model_frame), with_params = model_int)
        acc = mg.vali(network, mg.merge_data_iter(testing_data), device)
        acc_list.append(acc)
        if testing_data_B is not None:
            acc_B = mg.vali(network, mg.merge_data_iter(testing_data_B), device)
            acc_list_B.append(acc_B)
            
    fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
    axes = fig.subplots()
    # Plot with dashed lines and markers
    axes.plot(alpha_list, acc_list, 'o-', label="Accuracy on A", alpha = 0.6, color = "red")    
    if testing_data_B is not None:
        axes.plot(alpha_list, acc_list_B, 'o-', label="Accuracy on B", alpha = 0.6, color = "blue")    
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
    
    plt.savefig(os.path.join(log_plot_path, f"hypernet_1d_lmc_{title}.png" ), bbox_inches='tight') 
    plt.clf()   
    plt.close(fig) 