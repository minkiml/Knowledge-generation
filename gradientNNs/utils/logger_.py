import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# Set plt params
sns.set(style='ticks', font_scale=1.2)
plt.rcParams['figure.figsize'] = 12,8

class Logger(object):
    '''
    Handy logger object to log every training histories. 
    '''
    def __init__(self,
                 plot_path, # log path
                 hist_path,
                 *argw):
        self.log_plot_path = plot_path
        self.log_train_path = hist_path
        if argw is not None:
            # CSV
            self.types = []
            # -- print headers
            with open(os.path.join(self.log_train_path, "training_log.csv"), '+a') as f:
                for i, v in enumerate(argw, 1):
                    self.types.append(v[0])
                    if i < len(argw):
                        print(v[1], end=',', file=f)
                    else:
                        print(v[1], end='\n', file=f)

    def log_into_csv_(self, *argw):
        # logging losses, lr, etc, that are generated at every epoch or iteration into csv
        with open(os.path.join(self.log_train_path, "training_log.csv"), '+a') as f:
            for i, tv in enumerate(zip(self.types, argw), 1):
                end = ',' if i < len(argw) else '\n'
                print(tv[0] % tv[1], end=end, file=f)

    def log_stats_into_csv_(self, *argw):
        # logging losses, lr, etc, that are generated at every epoch or iteration into csv

        with open(os.path.join(self.log_train_path, "stats.csv"), '+a') as f:
            f.write(f"{argw[0].label}_mean,{argw[0].label}_std,{argw[0].label}_L2,{argw[1].label}_mean,{argw[1].label}_std,{argw[1].label}_L2,{argw[2].label}_mean,{argw[2].label}_std,{argw[2].label}_L2\n")
            for val1_mean, val1_std, val1_norm, val2_mean, val2_std, val2_norm, val3_mean, val3_std, val3_norm in zip(argw[0].mean_list, argw[0].std_list, argw[0].norm_list,
                                        argw[1].mean_list, argw[1].std_list, argw[1].norm_list,
                                        argw[2].mean_list, argw[2].std_list, argw[2].norm_list):
                f.write(f"{val1_mean.item()},{val1_std.item()},{val1_norm.item()},{val2_mean.item()},{val2_std.item()},{val2_norm.item()},{val3_mean.item()},{val3_std.item()},{val3_norm.item()}\n")
   
    # get more functions on demand
    def log_pics(self, x, y, name_ = "", epo_ = 0):
        # Save 2d scatters (rep)
        fig = plt.figure(figsize=(12, 8), 
          dpi = 600) 
        axes = fig.subplots()
        scatter = axes.scatter(x = x[:,0], y= x[:,1], c = y[:], 
                    s=15, cmap="Spectral", alpha = 0.8)# , edgecolors= "black" 

        # Get unique class labels
        unique_classes = np.unique(y[:, 0])

        # Iterate through each class and mark one point
        for ii, class_label in enumerate(unique_classes):
            class_indices = np.where(y[:, 0] == class_label)[0]
            sample_index = class_indices[10]  # Choose the first sample for marking
            axes.text(x[sample_index, 0], x[sample_index, 1], f'{ii}', color='black', fontsize=15, 
                      ha='center', va='center', fontweight='bold')

        cbar = plt.colorbar(scatter)
        cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=20)
        # color_bar = plt.colorbar()
        # color_bar.set_label('Domain Prediction', rotation=270, labelpad=20, fontsize=20)

        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        plt.savefig(os.path.join(self.log_plot_path,name_ + f"_{epo_}.png" )) 
        plt.clf()   
        plt.close(fig)

    # get more functions on demand
    def log_forecasting_vis(self, pred, ground_t, gt_ext = None, name_ = "", last_ = False):
        B, L, C = pred.shape
        c_ = 1 if last_ else C
        # assert pred.shape == ground_t.shape
        for i in range (c_ if c_ < 10 else 7):
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            axes.plot(pred[-1, :,i], color = "red",
                alpha = 0.8, label = 'NFM')
            axes.plot(ground_t[-1, :,i], color = "blue",
                alpha = 0.8, label = 'Ground truth')
            if gt_ext is not None:
                axes.plot(gt_ext[-1, :,i], color = "green",
                    alpha = 0.8, label = 'Ground truth full')
            plt.xticks(fontweight='bold', fontsize = 20)   
            plt.yticks(fontweight='bold', fontsize = 20)
            legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1))

            plt.savefig(os.path.join(self.log_plot_path, name_ + f"_feature {i}_" + ".png" ), bbox_inches='tight') 
            plt.clf()   
            plt.close(fig) 
    # get more functions on demand
    def log_forecasting_error_vis(self, errors):
        td_ = errors.T.detach().cpu().numpy()
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        im = axes.imshow(td_, cmap='viridis', aspect='auto')
        # Set axis labels and title
        plt.xlabel("seq")
        plt.ylabel("feature")
        # Add a colorbar using the ScalarMappable
        cbar = plt.colorbar(im)
        # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
        plt.savefig(os.path.join(self.log_plot_path, "Forecasting_Error.png")) 
        plt.clf()   
        plt.close(fig)

    def frequency_reponse(self, f = None, range = -1, name_ = ""):
        if f.dim() == 2:
            F, c = f.shape
            magnitude_response = 20 * np.log10(np.abs(f.cpu().detach().numpy()))
        else:
            B, F, c = f.shape
            magnitude_response = 20 * np.log10(np.abs(f.cpu().detach().numpy()) + 1e-4).mean(axis=0)
        magnitude_response[magnitude_response <= -3] = - 40
        fig = plt.figure(figsize=(12, 8), 
        dpi = 600) 
        axes = fig.subplots()
        # axes.plot(freq_bins, magnitude_response.mean(axis = 1) if c > 1 else magnitude_response[:,0], color='blue')
        im = axes.imshow(magnitude_response.T, aspect='auto', cmap='viridis')
        plt.title('Frequency response')
        plt.xlabel('Frequency Component (F)')
        plt.ylabel('Feature')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\mathbf{Gain (dB)}$', fontweight='bold', fontsize=25)
        # plt.xlim(0, freq_bins.shape[0] if range == -1 else range)
        # plt.grid()
        plt.savefig(os.path.join(self.log_plot_path,"Frequency_gain_" + f"{name_}" + ".png" )) 
        plt.clf()   
        plt.close(fig)
  
        fig = plt.figure(figsize=(12, 8), 
        dpi = 600) 
        axes = fig.subplots()
        # axes.plot(freq_bins, magnitude_response.mean(axis = 1) if c > 1 else magnitude_response[:,0], color='blue')
        axes.plot(magnitude_response.mean(1), color='blue')
        plt.title('Frequency response')
        plt.xlabel('Frequency Component (F)')
        plt.ylabel('Gain')
        plt.savefig(os.path.join(self.log_plot_path,"Frequency_gain1d_" + f"{name_}" + ".png" )) 
        plt.clf()   
        plt.close(fig)

    def log_confusion_matrix(self, pred, true, classes = None, name_ = ""):
        cm = confusion_matrix(true.detach().cpu().numpy(), pred.detach().cpu().numpy())
        fig = plt.figure(figsize=(12, 8), 
                dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        heatmap = sns.heatmap(cm, annot=True,fmt='d', cmap='viridis')
        

        # Set axis labels and title
        plt.xlabel("Prediction", fontsize=20, fontweight='bold')
        plt.ylabel("Ground Truth", fontsize=20, fontweight='bold')
        plt.xticks(fontsize = 20)   
        plt.yticks(fontsize = 20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_plot_path, "CM_" + f"{name_}" + ".png")) 
        plt.clf()   
        plt.close(fig)
        
    def vis_latent(self, z, cls_labels, superlabels = None):
        z = (z - z.mean(0)) / z.std()
        z = z.detach().cpu().numpy()
        pca = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=42)
        z = pca.fit_transform(z)
        # pca.fit(z)
        # z = pca.transform(z)
        if superlabels is not None:
            labels = [cls_labels.detach().cpu().numpy(), superlabels.detach().cpu().numpy()]
            labels_name = ["All classes", "Inter classes"]
        else:
            labels = [cls_labels.detach().cpu().numpy()]
            labels_name = ["All classes"]
        for lab, name in zip(labels, labels_name):
            fig = plt.figure(figsize=(12, 8), 
                dpi = 600) 
            axes = fig.subplots()
            
            scatter = axes.scatter(x = z[:,0], y= z[:,1], c =lab[:], 
                                    s=20, cmap="Spectral", alpha = 0.8, edgecolors= "black")
            cbar = plt.colorbar(scatter)
            cbar.set_label(name, fontweight='bold', fontsize=20)

            # cbar.set_label(r'$\mathbf{c}$', fontweight='bold', fontsize=20)
            
            plt.xlabel("Principal component 1", fontsize=20, fontweight='bold')
            plt.ylabel("Principal component 2", fontsize=20, fontweight='bold')
            plt.xticks(fontweight='bold', fontsize = 20)   
            plt.yticks(fontweight='bold', fontsize = 20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_plot_path,f"latent_{name}.png" )) 
            plt.clf()   
            plt.close(fig)
    def rul_plot_all(self, pred, y, title, sort = False, x_label ="Time"):
        y= y.view(-1)
        pred = pred.view(-1)
        t = torch.arange(y.shape[0])
        
        if sort:
            # Sort 
            sorted_indices = torch.argsort(y, dim=0)

            # Step 2: Apply the sorted indices to both B and A
            y = y[sorted_indices]
            pred = pred[sorted_indices]
        
        error = (y - pred).abs()
        
        fig, (ax_main, ax_err) = plt.subplots(
                                            2, 1,
                                            sharex=True,
                                            figsize=(12, 8),
                                            dpi=600,
                                            gridspec_kw={'height_ratios': [3, 1]}
                                                )
        # --- Top subplot: prediction vs ground truth ---
        ax_main.plot(y[:].detach().cpu().numpy(), c = "orange", linewidth=1.5, alpha=0.8, label = "Ground Truth", ls = "--")
        ax_main.plot(pred[:].detach().cpu().numpy(), c = "blue", linewidth=1., alpha=1.0, label = "Prediction") # , marker = "o"
        ax_main.set_ylabel("Remaining Useful Life", fontsize = 20, fontweight = 'bold')
        # ax_main.tick_params(axis='both', labelsize=20)
        # for label in ax_main.get_xticklabels() + ax_main.get_yticklabels():
        #     label.set_fontweight('bold')
        ax_main.legend(fontsize=20, 
                        loc='upper right',  # This is relative to the bbox
                            edgecolor='black', facecolor='white',
                        frameon=True,  # Ensures the frame is on
                        framealpha=1,  # Makes the frame completely opaque
                        fancybox=True,
                        ncol = 2)
        ax_main.grid(True)

        # --- Bottom subplot: error ---
        ax_err.fill_between(t, 0, error.detach().cpu().numpy(), color='limegreen', alpha=0.3, label='Absolute Error')
        ax_err.plot(error.detach().cpu().numpy(), color='green', linewidth=1)
        ax_err.set_ylabel("Error", fontsize = 20, fontweight = 'bold')
        ax_err.set_xlabel(x_label, fontsize = 20, fontweight = 'bold')
        # ax_err.set_ylim(0, 0.5)
        # ax_err.tick_params(axis='both', labelsize=20)
        # for label in ax_err.get_xticklabels() + ax_err.get_yticklabels():
        #     label.set_fontweight('bold')
        ax_err.legend(fontsize=20, 
                                     loc='upper right',  # This is relative to the bbox
                                     edgecolor='black', facecolor='white',
                                    frameon=True,  # Ensures the frame is on
                                    framealpha=1,  # Makes the frame completely opaque
                                    fancybox=True)
        ax_err.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_plot_path, f"{title}_rul.png" ))   
        plt.clf()   
        plt.close(fig)
        
    def raw_plot(self, x, title = ""):
        x = x[:,::10,:]
        x = x.view(-1,2).detach().cpu().numpy()
        fig = plt.figure(figsize=(12, 8), 
                            dpi = 600) 
        axes = fig.subplots()
        
        axes.plot(x[:,0], c = "blue", linewidth=1.5, alpha=0.8, label = "Ground Truth channel 1")
        axes.plot(x[:,1], c = "blue", linewidth=1.5, alpha=0.8, label = "Ground Truth channel 2")

        plt.xlabel("Time", fontsize=20, fontweight='bold')
        plt.ylabel("Remaining useful life", fontsize=20, fontweight='bold')
        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        legend = axes.legend(fontsize=20, loc='upper right', bbox_to_anchor=(0.5, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_plot_path, f"femto_{title}_raw.png" ))    
        plt.clf()   
        plt.close(fig)
        
    def anomaly_score_plot(self, x, y=None, title = ""):
        # colours = ["green", "blue", "orange"]
        # labels = ["TD_score", "FD_score", "True"]
        colours = ["blue",  "orange"]
        labels = ["Prediction", "True"]
        fig = plt.figure(figsize=(12, 8), 
                            dpi = 600) 
        axes = fig.subplots()
        lines = []
        for i, xx in enumerate(x):
            xx = xx.detach().cpu().numpy()
            line_, = axes.plot(xx[:], c = colours[i], linewidth=1.5, alpha=0.5, label = labels[i])
            lines.append(line_)
        if y is not None:
            start = None
            for j in range(len(y)):
                if y[j] == 1 and start is None:
                    start = j
                elif y[j] == 0 and start is not None:
                    axes.axvspan(start, j, color='red', alpha=0.3)
                    start = None
            # Handle case where B ends with 1s
            if start is not None:
                axes.axvspan(start, len(y), color='red', alpha=0.3)
            red_patch = Patch(facecolor='red', alpha=0.3, label='Anomaly')
            lines.append(red_patch)
        plt.xlabel("Samples", fontsize=20, fontweight='bold')
        plt.ylabel("Score", fontsize=20, fontweight='bold')
        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        legend = axes.legend(handles= lines, fontsize=20, loc='upper right', bbox_to_anchor=(0.5, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_plot_path, f"ad_score_{title}.png" ))    
        plt.clf()   
        plt.close(fig)
        
class Value_averager(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count        
    @property
    def _get_avg(self):
        return self.avg
    
def grad_logger(named_params, prob_ = 'linears'):
    stats = Value_averager()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if prob_ in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats

def grad_logger_spec(named_params, prob_ = 'linears', off = False):
    stats = Value_averager()
    stats.first_layer = None
    stats.last_layer = None
    if not off:
        for n, p in named_params:
            if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
                pass
                if prob_ in n:
                    grad_norm = float(torch.norm(p.grad.data))
                    stats.update(grad_norm)
    return stats

class param_stats_tracker(object):
    def __init__(self, name_= ""):
        self.reset(name_)
    def reset(self, name_ ):
        self.norm_ = 0.
        self.mean = 0
        self.std= 0

        self.mean_history = []
        self.std_history = []
        self.norm_history = []
        self.label = name_
    def update(self, p_, n=1):
        try:
            self.mean = torch.mean(p_)
            self.std = torch.std(p_)
            self.norm_ = torch.norm(p_)

            self.mean_history.append(self.mean)
            self.std_history.append(self.std)
            self.norm_history.append(self.norm_)
        except Exception:
            pass 
    def logging_(self, named_params):
        prob_ = self.label
        for n, p in named_params:
            if not (n.endswith('.bias') or len(p.shape) == 1):
                if prob_ in n:
                    self.update(p.data)
    def out_logged(self):
        self.mean_list = torch.stack((self.mean_history), dim = 0)
        self.std_list = torch.stack((self.std_history), dim = 0)
        self.norm_list = torch.stack((self.norm_history), dim = 0)

