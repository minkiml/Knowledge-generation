'''
https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane.py

'''
import os
import logging
import time
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from tqdm import tqdm
import merging as mg
# from SLMC.trains.validation import vali
# from dataset.dataload_ import Load_dataset
# from SLMC.subspace import Subspace_g
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import warnings
import seaborn as sns; sns.set()
# Set plt params
sns.set(style='ticks', font_scale=1.2)
plt.rcParams['figure.figsize'] = 12,8

warnings.filterwarnings('ignore', category=FutureWarning)

def w_to_xy(p, w0, u, v):
    return np.array([np.dot(p - w0, u), np.dot(p - w0, v)])

def xy_to_w(x, y, 
            w0, u, v, dx, dy):
    return w0 + x * dx * u + y * dy * v

def get_u_v(w0, w1, w2):
    # w0 origin (e.g., known point = origin of subspace init)
    # w1 (e.g., known one end point)
    # w2 (e.g., known anotehr end point)
    # Note that w0,w1 and w2 must not be in the same line 
    u = w2 - w0
    dx = np.linalg.norm(u)
    u /= dx

    v = w1 - w0
    v -= np.dot(u, v) * u
    dy = np.linalg.norm(v)
    v /= dy
    return u, v, dx, dy

def list_params_to_vector(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()]).detach()

def vector_params_to_list(model, vector_w):
    pointer = 0
    vector_w = torch.from_numpy(vector_w)
    for param in model.parameters():
        # Get the number of elements in this parameter
        num_param = param.numel()
        # Slice the vector and reshape to the param's shape
        new_param = vector_w[pointer:pointer + num_param].view_as(param)
        # Overwrite the param data
        param.data.copy_(new_param)
        pointer += num_param
    return model

def construct_loss_surface(args,
                       model_A, model_B, init_model,
                       testing_data,
                       device = None,
                    #    limit_batch = 0, # Limit mini-batch for eval to speed up.
                       
                       # plane arguments
                       grid_dim =21, # plane is of grid by grid (determines resolution)
                       margin = [0.2, 0.2, 0.2, 0.2], # left right bottom top
                       num_from_path = 30, # in case we have a modelled path, we could sample from it.  

                       save_results = True,
                       save_name = "",
                       **additional_models
                       ):

    # Domain over which plane is defined
    range_ = [0., 1.]

    # Define plane of grid_dim by grid_dim
    x_axis = np.linspace(range_[0] - margin[0], range_[1] + margin[1], grid_dim)
    y_axis = np.linspace(range_[0] - margin[2], range_[1] + margin[3], grid_dim)

    # Initialize lists to store info
    losses = np.zeros([grid_dim, grid_dim], dtype=np.float32)
    accs = np.zeros([grid_dim, grid_dim], dtype=np.float32)
    grid = np.zeros((grid_dim, grid_dim, 2))
 
    # Set w0, w1, and w2
    # We set w0 (initiali theta), w1 (one end point), w2 (another end point)    
    # get point-network params and vectorizing into w0, w1, w2
    
    w0 = list_params_to_vector(init_model).cpu().numpy().ravel()
    w1 = list_params_to_vector(model_A).cpu().numpy().ravel()
    w2 = list_params_to_vector(model_B).cpu().numpy().ravel()

    # Get edge parameters of plane
    u, v, dx, dy = get_u_v(w0, w1, w2)


    # Get coordinates of w0 w1 w2 --> save them in the order of w1 w2 w0
    edge_coordinates = np.stack([w_to_xy(p, w0, u, v) for p in [w1, w2, w0]])
    if additional_models is not None:
        model_D = additional_models["Model_D"]
        w3 = list_params_to_vector(model_D).cpu().numpy().ravel()
        w3_coor = w_to_xy(w3, w0, u, v)
    else:
        w3_coor = None
    # Sample theta between w1 and w2 (only linear path is available for now)
    # Linear interpolation line between w1 and w2
    path_coordinates = []
    alpha_list = torch.linspace(0, 1, steps=num_from_path)
    for alpha in alpha_list:
        path_coordinates.append(w_to_xy((alpha * w1) + ((1 - alpha) * w2), w0, u, v))
    path_coordinates = np.stack(path_coordinates)

    # get accuracy for all coordinates in defined grid
    with torch.no_grad(): 
        for i, x in tqdm(enumerate(x_axis), total = len(x_axis)):
            for j, y in enumerate(y_axis):
                # Get w of the corresponding coordinates (x, y) 
                p = xy_to_w(x, y,
                            w0, u, v, dx, dy)
                # Assignment set model param to this 
                model = vector_params_to_list(init_model, p).to(device)
                # Run eval set model param to this
                acc, loss = mg.vali(model, mg.merge_data_iter(testing_data), device, yield_loss = True)
                grid[i, j] = [x * dx, y * dy]
                accs[i, j] = acc
                losses[i, j] = loss
             
    if save_results:
        np.savez(
            os.path.join(args.his_save_path, f'plane_inf_{save_name}.npz'),
            edge_coordinates=edge_coordinates,
            path_coordinates=path_coordinates,
            x_axis=x_axis,
            y_axis=y_axis,
            grid=grid,
            accs=accs,
            losses=losses,
            w3_coor=w3_coor
        )


def plot_loss_surface(args,
                      save_name = "",
                      plot_ps_together = False):

    # Load 
    try:
        loss_surface_data = np.load(os.path.join(args.his_save_path, f'plane_inf_{save_name}.npz'))
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    else:

        grid, accs, losses = loss_surface_data['grid'], loss_surface_data['accs'], loss_surface_data['losses']

        edge_coords, path_coords = loss_surface_data['edge_coordinates'], loss_surface_data['path_coordinates']
        
        if loss_surface_data['w3_coor'] is not None:
            w3_coor = loss_surface_data['w3_coor']
        else: w3_coor = None
        all_loss = [losses]
        all_acc = [accs]
        # if accs_s != "None":
        #     # Combine two loss landscape by collecting max or min 
        #     if plot_ps_together:
        #         all_loss = [np.minimum(losses, losses_s)]
        #         all_acc = [np.maximum(accs, accs_s)]
        #     else:
        #         all_loss.append(losses_s)
        #         all_acc.append(accs_s)

        for jj, (losses, accs) in enumerate(zip(all_loss, all_acc)):
            if plot_ps_together:
                function_title = r'$f_p, f_s$'
            else:
                if jj == 0:
                    function_title = r'$f_p$'
                else:
                    function_title = r'$f_s$'
            contents = [losses, accs]
            title = ["Loss", "ACC"]
            vmaxs = [4.5, 100]
            log_alphas = [-4.0, -0.5] #[-2.9, -0.005] # smaller the alpha is, wider the contour range (i.e., more unification)
            N=10
            for ii, (values, vmax, log_alpha, t) in enumerate(zip(contents, vmaxs , log_alphas, title)):
                if ii == 1:
                    values = 100.0 - values  
                    # vmax *= 0.1
                # print(t, values)
                if vmax is None:
                    clipped = values.copy()
                else:
                    clipped = np.minimum(values, vmax)
                log_gamma = (np.log(clipped.max() - clipped.min()) - log_alpha) / N
                levels = clipped.min() + np.exp(log_alpha + log_gamma * np.arange(N + 1))

                levels[0] = clipped.min()
                levels[-1] = clipped.max()
                norm = LogNormalize(clipped.min() - 1e-8, clipped.max() + 1e-8, log_alpha=log_alpha)

                fig = plt.figure(figsize=(12, 8), 
                        dpi = 600) 
                axes = fig.subplots()

                contour = axes.contour(grid[:, :, 0], grid[:, :, 1], values, cmap="turbo" if ii==0 else "viridis", norm=norm,
                                    linewidths=1.5,
                                    zorder=1,
                                    levels=levels,
                                    extend="max")
                contourf = axes.contourf(grid[:, :, 0], grid[:, :, 1], values, cmap="turbo" if ii==0 else "viridis", norm=norm,
                                        levels=levels,
                                        zorder=0,
                                        alpha=0.5,
                                        extend="max")
                
                plt.colorbar(contourf)
                
                # axes.scatter(edge_coords[[0, 2], 0], edge_coords[[0, 2], 1], marker='o', c='k', s=120, zorder=2)
                axes.scatter(edge_coords[0, 0], edge_coords[0, 1], marker='o', c='k', s=120, zorder=2)
                axes.scatter(edge_coords[1, 0], edge_coords[1, 1], marker='D', c='k', s=120, zorder=2)
                axes.scatter(edge_coords[2, 0], edge_coords[2, 1], marker='D', s=120, zorder=2, color = "green")
                if w3_coor is not None:
                    axes.scatter(w3_coor[0], w3_coor[1], marker='D', s=120, zorder=2, color = "black")
                # if dis_coords != "None":
                #     axes.scatter(dis_coords[0], dis_coords[1], marker='D', s=120, zorder=2, color = "blue")


                axes.plot(path_coords[:, 0], path_coords[:, 1], linewidth=2.5, linestyle='--', c='k', label=r'$P(\alpha)$', zorder=4)
                # axes.plot(edge_coords[[0, 1], 0], edge_coords[[0, 1], 1], c='k', linestyle='--', dashes=(3, 4), linewidth=3, zorder=2)

                plt.annotate(r'$\theta^{*}_{train}$', [edge_coords[0, 0], edge_coords[0, 1]], color='black',
                            size=20, weight='bold')
                plt.annotate(r'$\theta^{*}_{full}$', [edge_coords[1, 0], edge_coords[1, 1]], color='black',
                            size=20, weight='bold')
                plt.annotate(r'$\theta^{init}$', [edge_coords[2, 0], edge_coords[2, 1]], color='green',
                            size=20, weight='bold')
                if w3_coor is not None:
                    plt.annotate(r'$\theta^{test}$', [w3_coor[0], w3_coor[1]], color='black',
                            size=20, weight='bold')
                # if dis_coords != "None":
                #     plt.annotate(r'$\theta^{dist}_{g \rightarrow s}$', [dis_coords[0], dis_coords[1]], color='blue',
                #             size=20, weight='bold')
                    
                plt.yticks(fontsize=20, fontweight='bold')
                plt.xticks(fontsize=20, fontweight='bold')
                # plt.title(f'{t} landscape ({function_title})', fontsize=20)
                plt.savefig(os.path.join(args.plots_save_path,f"{t}_{save_name}_surface {jj if not plot_ps_together else 2}" + ".png" ), bbox_inches='tight') 
                plt.clf()   
                plt.close(fig)

class LogNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)
        
    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)


