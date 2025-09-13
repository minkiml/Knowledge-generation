import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# Set plt params
plt.rcParams['agg.path.chunksize'] = 1000
sns.set(style='ticks', font_scale=1.2)
plt.rcParams['figure.figsize'] = 12,8

class Logger(object):
    '''
    Handy logger object to log every training histories. 
    '''
    def __init__(self,
                 plot_path, # log path
                 hist_path,
                 file_name,
                 *argw):
        self.log_plot_path = plot_path
        self.log_train_path = hist_path
        self.file_name = file_name
        if argw is not None:
            # CSV
            self.types = []
            self.types_acc = []
            # -- print headers
            with open(os.path.join(self.log_train_path, f"training_log_{file_name}.csv"), '+a') as f:
                for i, v in enumerate(argw[:-2], 1):
                    self.types.append(v[0])
                    if i < len(argw):
                        print(v[1], end=',', file=f)
                    else:
                        print(v[1], end='\n', file=f)
            with open(os.path.join(self.log_train_path, f"accs_{file_name}.csv"), '+a') as f:
                for i, v in enumerate(argw[-2:], 1):
                    self.types_acc.append(v[0])
                    if i < len(argw):
                        print(v[1], end=',', file=f)
                    else:
                        print(v[1], end='\n', file=f)

    def log_into_csv_(self, *argw):
        # logging losses, lr, etc, that are generated at every epoch or iteration into csv
        with open(os.path.join(self.log_train_path, f"training_log_{self.file_name}.csv"), '+a') as f:
            for i, tv in enumerate(zip(self.types, argw), 1):
                end = ',' if i < len(argw) else '\n'
                print(tv[0] % tv[1], end=end, file=f)

    def log_acc_into_csv_(self, *argw):
        # logging losses, lr, etc, that are generated at every epoch or iteration into csv

        with open(os.path.join(self.log_train_path, f"accs_{self.file_name}.csv"), '+a') as f:
            for i, tv in enumerate(zip(self.types_acc, argw), 1):
                end = ',' if i < len(argw) else '\n'
                print(tv[0] % tv[1], end=end, file=f)
   
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
            axes.text(x[sample_index, 0], x[sample_index, 1], f'{ii}', color='black', fontsize=15, ha='center', va='center', fontweight='bold')

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
    def log_pred_vis(self, x, epo):
        td_ = x.T.detach().cpu().numpy()
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        im = axes.imshow(td_, cmap='viridis', aspect='auto')
        # Set axis labels and title
        plt.xlabel("depth")
        plt.ylabel("class")
        # Add a colorbar using the ScalarMappable
        cbar = plt.colorbar(im)
        # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
        plt.savefig(os.path.join(self.log_plot_path, f"depth_vs_class_{epo}.png")) 
        plt.clf()   
        plt.close(fig)
    def log_pre_on_off(self, pred, acc):
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        ax = fig.subplots()
        cax = ax.matshow(pred.T, cmap='viridis')  # Transposing the data for the desired orientation

        # Setting the axis labels and ticks
        ax.set_ylabel('Case')
        ax.set_xlabel('Samples')
        ax.set_yticks(np.arange(len(acc)))
        ax.set_yticklabels(np.arange(1, len(acc) + 1), fontsize=8)  # Adjust font size
        ax.set_xticks(np.arange(0, 32, 2))  # Adjusted tick spacing
        ax.set_xticklabels(np.arange(0, 32, 2), fontsize=8)  # Adjust font size

        # Adding a color bar
        cbar = fig.colorbar(cax)
        cbar.set_label('Class')

        # Marking elements of class 10 as "off" on the matrix plot
        for x in range(pred.shape[0]):
            for y in range(pred.shape[1]):
                if pred[x, y] == 10:
                    ax.text(x, y, 'off', va='center', ha='center', color='red', fontsize=8)

        # Adjusting the annotation box for quantities to be more visible and away from the plot
        quantities_str1 = ", ".join([f"{i+1}: {q}" for i, q in enumerate(acc[:5])])
        quantities_str2 = ", ".join([f"{i+6}: {q}" for i, q in enumerate(acc[5:])])
        full_quantities_str = f"{quantities_str1}\n{quantities_str2}"
        ax.annotate(full_quantities_str, xy=(0.5, 1), xytext=(0, 20),  # Increasing the offset for more space
                    xycoords='axes fraction', textcoords='offset points',
                    size=10, ha='center', va='baseline', weight='bold')  # Making text bold and slightly larger

        plt.savefig(os.path.join(self.log_plot_path, f"test_2.png")) 
        plt.clf()   
        plt.close(fig)

    def log_acc_vis(self, vis_,epo):

        fig = plt.figure(figsize=(12, 8), 
        dpi = 600) 
        axes = fig.subplots()
        axes.plot(vis_, color = "red")
        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        plt.savefig(os.path.join(self.log_plot_path, f"acc_all_{epo}" + ".png" ), bbox_inches='tight') 
        plt.clf()   
        plt.close(fig) 

    # get more functions on demand
    def log_feature_vis(self, x, epo):
        if x.shape[0] > 1:
            for ii in range(x.shape[0]):
                td_ = x[ii].T.detach().cpu().numpy()
                fig = plt.figure(figsize=(12, 8), 
                    dpi = 600) 
                axes = fig.subplots()
                # Display the attention map
                im = axes.imshow(td_, cmap='viridis', aspect='auto')
                # Set axis labels and title
                plt.xlabel("depth")
                plt.ylabel("k-net_features")
                # Add a colorbar using the ScalarMappable
                cbar = plt.colorbar(im)
                # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
                plt.savefig(os.path.join(self.log_plot_path, f"depth_vs_Kfeature_{epo}_{ii}.png")) 
                plt.clf()   
                plt.close(fig)
        
    # get more functions on demand
    def log_filter_vis(self, x, epo, number = 16):
        num = int(np.sqrt(number))
        td_ = x.detach().cpu().numpy()
        fig, axes = plt.subplots(num, num, figsize=(12, 12), dpi=600)
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(td_[i], cmap='viridis', aspect='auto')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        # Add a colorbar using the ScalarMappable
        # cbar = plt.colorbar(im)
        # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
        plt.savefig(os.path.join(self.log_plot_path, f"filter_epoch{epo}.png")) 
        plt.clf()   
        plt.close(fig)
    
    def log_weights_vis(self, x, epo, depth = 0, progressive = False):
        # x (depth, d, d)
        if x.dim() == 3:
            depth_ = x.shape[0]
            if depth_> 10:
                depth_ = 10
            for i in range(depth_):
                if progressive:
                    progressive_flag = "progressive"
                    if i == 0:
                        td_ = ((x[i] - x[0])**2).T.detach().cpu().numpy()
                    else:
                        td_ = ((x[i] - x[i-1])**2).T.detach().cpu().numpy()
                else:
                    progressive_flag = ""
                    td_ = x[i].T.detach().cpu().numpy()
                fig = plt.figure(figsize=(12, 8), 
                    dpi = 600) 
                axes = fig.subplots()
                # Display the attention map
                im = axes.imshow(td_, cmap='viridis', aspect='auto')
                # Set axis labels and title
                plt.xlabel("out")
                plt.ylabel("in")
                # Add a colorbar using the ScalarMappable
                cbar = plt.colorbar(im)
                # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
                plt.savefig(os.path.join(self.log_plot_path, f"{epo}_in_vs_out_depth{i}_{progressive_flag}.png")) 
                plt.clf()   
                plt.close(fig)
        else:
            if progressive:
                pass
            else:
                td_ = x.T.detach().cpu().numpy()
                fig = plt.figure(figsize=(12, 8), 
                    dpi = 600) 
                axes = fig.subplots()
                # Display the attention map
                im = axes.imshow(td_, cmap='viridis', aspect='auto')
                # Set axis labels and title
                plt.xlabel("out")
                plt.ylabel("in")
                # Add a colorbar using the ScalarMappable
                cbar = plt.colorbar(im)
                # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
                plt.savefig(os.path.join(self.log_plot_path, f"{epo}_in_vs_out_depth{depth}.png")) 
                plt.clf()   
                plt.close(fig)
    def log_time_domain_vis(self, ground_t, name_ = ""):
        fig = plt.figure(figsize=(12, 8), 
        dpi = 600) 
        axes = fig.subplots()
        axes.plot(ground_t[:,-1], color = "blue",
            alpha = 0.8, label = 'Input X')
        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1))

        plt.savefig(os.path.join(self.log_plot_path, name_ + f"Input X_{name_}" + ".png" ), bbox_inches='tight') 
        plt.clf()   
        plt.close(fig) 
        # get more functions on demand
    def frequency_reponse(self, in_, label_ = ""):
        
        magnitude = np.abs(in_.cpu().detach().numpy())
        # power spectrum
        magnitude_response = 20 * np.log10(magnitude)
        # torch.clamp(magnitude_response, min=0., max=None)
        if len(magnitude) > 200000:
            magnitude = np.stack(np.array_split(magnitude, 5))
            magnitude_response = np.stack(np.array_split(magnitude_response, 5))
            flag_ = True
        else: flag_ = False
        # axes.plot(freq_bins, magnitude_response.mean(axis = 1) if c > 1 else magnitude_response[:,0], color='blue')
        if not flag_:
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.plot(magnitude, color='b')
            plt.title('Frequency response')
            plt.xlabel('Frequency mode')
            plt.ylabel('Feature')

            plt.savefig(os.path.join(self.log_plot_path,"Frequency (gp_gs)" + f"{label_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)
    

            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            # axes.plot(freq_bins, magnitude_response.mean(axis = 1) if c > 1 else magnitude_response[:,0], color='blue')
            axes.plot(magnitude_response, color='b')
            plt.title('Frequency response')
            plt.xlabel('Frequency mode')
            plt.ylabel('Gain')

            plt.savefig(os.path.join(self.log_plot_path,"Power_spectrum (gp_gs)" + f"{label_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.imshow(magnitude.T, cmap='viridis', aspect='auto')
            plt.title('Frequency response')
            plt.xlabel('Frequency mode')
            plt.ylabel('Feature')
            cbar = plt.colorbar(im)
            plt.savefig(os.path.join(self.log_plot_path,"Frequency (gp_gs)" + f"{label_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)
    

            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            # axes.plot(freq_bins, magnitude_response.mean(axis = 1) if c > 1 else magnitude_response[:,0], color='blue')
            im2 = axes.plot(magnitude_response.T, cmap='viridis', aspect='auto')
            plt.title('Frequency response')
            plt.xlabel('Frequency mode')
            plt.ylabel('Gain')
            cbar = plt.colorbar(im2)
            plt.savefig(os.path.join(self.log_plot_path,"Power_spectrum (gp_gs)" + f"{label_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)


    def log_confusion_matrix(self, cm, name_ = ""):
        fig = plt.figure(figsize=(12, 8), 
                dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        sns.heatmap(cm, cmap='viridis')
        # Set axis labels and title
        plt.xlabel("GT")
        plt.ylabel("Pred")
        plt.savefig(os.path.join(self.log_plot_path, "CM_" + f"{name_}" + ".png")) 
        plt.clf()   
        plt.close(fig)
    

    def log_2d_vis(self, x, f = None, name_ = "", time = False):
        if time == False:
            if x.dim() == 2:
                l, c = x.shape
                # time domain
                td_ = x.T.detach().cpu().numpy()
                # mag
                magnitude = torch.abs(f).cpu().detach().numpy()
                # phase
                phase = torch.angle(f).cpu().detach().numpy()
                # freq (= mag)
                imaginary =(f.imag).cpu().detach().numpy()
                real_ =(f.real).cpu().detach().numpy()
            else:
                b, l, c = x.shape
                td_ = x.mean(dim=0).T.detach().cpu().numpy()
                magnitude = torch.abs(f).mean(dim=0).cpu().detach().numpy()
                phase = torch.angle(f).mean(dim=0).cpu().detach().numpy()
                imaginary =(f.imag).mean(dim=0).cpu().detach().numpy()
                real_ =(f.real).mean(dim=0).cpu().detach().numpy()
            fig = plt.figure(figsize=(12, 8), 
                dpi = 600) 
            axes = fig.subplots()
            # Display the attention map
            im = axes.imshow(td_, cmap='viridis', aspect='auto')
            # Set axis labels and title
            plt.xlabel("t")
            plt.ylabel("feature")
            # Add a colorbar using the ScalarMappable
            cbar = plt.colorbar(im)
            # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
            plt.savefig(os.path.join(self.log_plot_path, "LFT_" + f"{name_}" + ".png")) 
            plt.clf()   
            plt.close(fig)

            # Magnitude
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.imshow(magnitude.T, aspect='auto', cmap='viridis')
            # plt.title('Magnitude Visualization')
            plt.xlabel('Frequency Component', fontweight='bold')
            plt.ylabel('Hidden Dimension', fontweight='bold')
            plt.xticks(fontweight='bold')   
            plt.yticks(fontweight='bold')
            cbar = plt.colorbar(im)
            cbar.set_label('Magnitude', fontweight='bold')
            # cbar.set_ticks(fontweight='bold')

            plt.savefig(os.path.join(self.log_plot_path,"magnitude_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)

            # Phase
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.imshow(phase.T, aspect='auto', cmap='twilight')
            plt.title('Phase Visualization')
            plt.xlabel('Frequency Component (F)')
            plt.ylabel('Feature Dimension (d)')
            cbar = plt.colorbar(im)
            plt.savefig(os.path.join(self.log_plot_path,"Phase_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)

            # Imaginary only
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.imshow(imaginary.T, aspect='auto', cmap='twilight')
            plt.title('Imaginary Values')
            plt.xlabel('Frequency Component (F)')
            plt.ylabel('Feature Dimension (d)')
            cbar = plt.colorbar(im)
            plt.savefig(os.path.join(self.log_plot_path,"Imaginary_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)

            # real only
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.imshow(real_.T, aspect='auto', cmap='twilight')
            plt.title('Real Values')
            plt.xlabel('Frequency Component (F)')
            plt.ylabel('Feature Dimension (d)')
            cbar = plt.colorbar(im)
            plt.savefig(os.path.join(self.log_plot_path,"Real_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)

            # Frequency
            ff_ = magnitude.mean(axis = 1)
            ff_[0] *= 0.
            freq_bins = np.arange(magnitude.shape[0])
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            axes.plot(freq_bins, ff_ if c > 1 else magnitude[:,0], color='blue')
            # plt.title('Frequency Domain Plot')
            plt.xlabel('Frequency Component', fontweight='bold')
            plt.ylabel('Magnitude', fontweight='bold')
            plt.xticks(fontweight='bold')   
            plt.yticks(fontweight='bold')
            plt.grid()
            plt.savefig(os.path.join(self.log_plot_path,"Mean_Frequency_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)
        else:
            td_ = x.mean(dim=0).T.detach().cpu().numpy()
            fig = plt.figure(figsize=(12, 8), 
                dpi = 600) 
            axes = fig.subplots()
            # Display the attention map
            im = axes.imshow(td_, cmap='viridis', aspect='auto')
            # Set axis labels and title
            plt.xlabel("t")
            plt.ylabel("feature")
            # Add a colorbar using the ScalarMappable
            cbar = plt.colorbar(im)
            # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
            plt.savefig(os.path.join(self.log_plot_path, "LFT_" + f"{name_}" + ".png")) 
            plt.clf()   
            plt.close(fig)

    def plot_interp_acc(self, alpha_list, acc_p_list, acc_s_list = None, 
                        mode = "sub", dataset = "MNIST", arch = "LeNet",
                        save_name = ""):
        if mode == "sub":
            title_ = f"Interpolation in subspace ({dataset}\{arch})"
            x_tick_0 = r'$\theta_{g \rightarrow s}$'
            x_tick_1 = r'$\theta_{g \rightarrow p}$'
        elif mode == "ori":
            title_ = f"Interpolation in original space ({dataset}\{arch})"
            x_tick_0 = r'$\theta_{g \rightarrow s}$'
            x_tick_1 = r'$\theta_{g \rightarrow p}$'
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        # Plot with dashed lines and markers
        axes.plot(alpha_list, acc_p_list, 'r-o', label=r'$f_p$', alpha = 0.6)
        if acc_s_list is not None:
            axes.plot(alpha_list, acc_s_list, 'g--s', label=r'$f_s$', alpha = 0.6)
        
        # Set title and labels
        plt.title(title_, fontsize=22)
        plt.xlabel(r'$\alpha$', fontsize=20)
        plt.ylabel('ACC (%)', fontsize=20)
        
        # Set x-ticks and labels
        xticks_positions = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        xticks_labels = [r'$\theta_{g \rightarrow s}$', '0.2', '0.4', '0.6', '0.8', r'$\theta_{g \rightarrow p}$']
        
        plt.xticks(ticks=xticks_positions, labels=xticks_labels, fontsize=20, fontweight='bold')
        
        # Customize colors for specific ticks
        for tick in axes.get_xticklabels():
            if tick.get_text() == x_tick_0:
                tick.set_color('purple')  # Set custom color for model A and model B
            if tick.get_text() == x_tick_1:
                tick.set_color('blue') 
        # Set y-ticks from 0 to 100 with intervals of 25
        yticks_positions = np.arange(0, 101, 25)
        plt.yticks(ticks=yticks_positions, fontsize=20, fontweight='bold')
        
        # Set legend with bold labels
        legend = plt.legend(loc='upper right', fontsize=18)
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # Find the indices of the best values in acc_p_list and acc_s_list
        best_p_idx = np.argmax(acc_p_list)
        if acc_s_list is not None:
            best_s_idx = np.argmax(acc_s_list)

        # Get corresponding alpha values for best points
        best_p_alpha = alpha_list[best_p_idx]
        if acc_s_list is not None:
            best_s_alpha = alpha_list[best_s_idx]

        # Annotate the best values
        axes.annotate(f'Best: {acc_p_list[best_p_idx]:.2f}%',
                    xy=(best_p_alpha, acc_p_list[best_p_idx]),
                    xytext=(best_p_alpha, acc_p_list[best_p_idx] + 2),
                    textcoords='data', ha='center', fontsize=14, color='red',
                    arrowprops=dict(facecolor='red', arrowstyle="->"))
        if acc_s_list is not None:
            axes.annotate(f'Best: {acc_s_list[best_s_idx]:.2f}%',
                        xy=(best_s_alpha, acc_s_list[best_s_idx]),
                        xytext=(best_s_alpha, acc_s_list[best_s_idx] + 2),
                        textcoords='data', ha='center', fontsize=14, color='green',
                        arrowprops=dict(facecolor='green', arrowstyle="->"))

        plt.savefig(os.path.join(self.log_plot_path, f"{save_name}.png" ), bbox_inches='tight') 
        plt.clf()   
        plt.close(fig) 

    def log_feature_emb(self, x, name):
        td_ = x.squeeze(0).detach().cpu().numpy()
        
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        im = axes.imshow(td_, cmap='viridis', aspect='auto')
        # Set axis labels and title
        plt.xlabel("W")
        plt.ylabel("H")
        # Add a colorbar using the ScalarMappable
        cbar = plt.colorbar(im)
        # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
        plt.savefig(os.path.join(self.log_plot_path, f"embeddings_hypernet_{name}.png")) 
        plt.clf()   
        plt.close(fig)
def plot_interp_acc_NNmerge( alpha_list, acc_list, save_name = "",
                            log_plot_path = None):

    # title_ = f"Interpolation in original space ({dataset}\{arch})"
    x_tick_0 = r'$\theta_{old}$' # #
    x_tick_1 = r'$\theta_{merged}$' # #
    fig = plt.figure(figsize=(12, 8), 
        dpi = 600) 
    axes = fig.subplots()
    # Plot with dashed lines and markers
    axes.plot(alpha_list, acc_list, 'r-o', label=r'$Interpolation$', alpha = 0.6) # #
    
    # Set title and labels
    # plt.title(title_, fontsize=22)
    plt.xlabel(r'$\alpha$', fontsize=20)
    plt.ylabel('ACC (%)', fontsize=20)
    
    # Set x-ticks and labels
    xticks_positions = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    xticks_labels = [r'$\theta_{old}$', '0.2', '0.4', '0.6', '0.8', r'$\theta_{merged}$'] # #
    
    plt.xticks(ticks=xticks_positions, labels=xticks_labels, fontsize=20, fontweight='bold')
    
    # Customize colors for specific ticks
    for tick in axes.get_xticklabels(): # #
        if tick.get_text() == x_tick_0:
            tick.set_color('purple')  # Set custom color for model A and model B
        if tick.get_text() == x_tick_1:
            tick.set_color('blue') 

    plt.ylim(bottom=10.0)

    # Set y-ticks from 0 to 100 with intervals of 25
    yticks_positions = np.arange(10, 101, 20)
    plt.yticks(ticks=yticks_positions, fontsize=20, fontweight='bold')
    
    # Set legend with bold labels
    legend = plt.legend(loc='lower right', fontsize=18)
    for text in legend.get_texts():
        text.set_fontweight('bold')



    # # Find the indices of the best values in acc_p_list and acc_s_list
    # best_p_idx = np.argmax(acc_list)
    # # Get corresponding alpha values for best points
    # best_p_alpha = alpha_list[best_p_idx]

    # # Annotate the best values
    # axes.annotate(f'Best: {acc_list[best_p_idx]:.2f}%',
    #             xy=(best_p_alpha, acc_list[best_p_idx]),
    #             xytext=(best_p_alpha, acc_list[best_p_idx] + 2),
    #             textcoords='data', ha='center', fontsize=14, color='red',
    #             arrowprops=dict(facecolor='red', arrowstyle="->"))
    
    # Annotate the first and last accuracy values
    first_acc = acc_list[0]
    last_acc = acc_list[-1]

    # Annotate the first accuracy
    axes.annotate(f'{first_acc:.2f}%', 
                  xy=(alpha_list[0], first_acc), 
                  xytext=(alpha_list[0], first_acc - 3),  # Positioning above the point
                  textcoords='data', ha='center', fontsize=14, color='green',
                  arrowprops=dict(facecolor='green', arrowstyle="->"))

    # Annotate the last accuracy
    axes.annotate(f'{last_acc:.2f}%', 
                  xy=(alpha_list[-1], last_acc), 
                  xytext=(alpha_list[-1], last_acc - 3),  # Positioning below the point
                  textcoords='data', ha='center', fontsize=14, color='blue',
                  arrowprops=dict(facecolor='blue', arrowstyle="->"))
    

    plt.savefig(os.path.join(log_plot_path, f"{save_name}.png" ), bbox_inches='tight') 
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

def grad_logger_spec(named_params, prob_ = 'linears', off = True):
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

class ModelSaver:
    def __init__(self, patience=7, verbose=False, logger = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = -np.Inf

        self.best_model = None
        self.early_stop = False
        self.logger = logger

    def __call__(self, acc, model, path, label = "ACC", model_name = None):
        if acc > self.best_acc:
            self.save_checkpoint(acc, model, path, label = label, model_name = model_name)
            self.best_acc = acc
            self.counter = 0
        else:
            if self.patience != -1:
                self.counter += 1
                self.logger.info(f'No validation improvement & EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, acc, model, path,
                        label = "ACC", model_name = None):
        if self.verbose:
            self.logger.info(f'Validation improved ({label}: {np.abs(self.best_acc):.6f} --> {np.abs(acc):.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, '_checkpoint.pth' if model_name is None else f'_checkpoint_{model_name}.pth'))
        self.best_model = model.state_dict()

    def get_best_model(self, model):
        assert self.best_model is not None
        return model.load_state_dict(self.best_model)

class ModelSaver_itr:
    def __init__(self, patience=7, verbose=True, logger = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc_p = -np.Inf
        self.best_acc_s = -np.Inf

        self.best_model = None
        self.early_stop = False
        self.logger = logger

    def __call__(self, acc, model, path, epo = 0, save = False):
        acc[0] = 0. if acc[0] is None else acc[0]
        acc[1] = 0. if acc[1] is None else acc[1]
        self.save_checkpoint(acc, model, path, epo = epo, save_= save)
        if acc[0] > self.best_acc_p:
            self.best_acc_p = acc[0]

        if acc[1] > self.best_acc_s:
            self.best_acc_s = acc[1]

        self.counter = 0

    def save_checkpoint(self, acc, model, path,
                        epo = 0, save_ = False):
        if self.verbose:
            self.logger.info(f'Validation results - ACC_p: best ({np.abs(self.best_acc_p):.6f}%) and current ({np.abs(acc[0]):.6f}%). ')
            self.logger.info(f'Validation results - ACC_s: best ({np.abs(self.best_acc_s):.6f}%) and current ({np.abs(acc[1]):.6f}%). ')
        if save_:
            torch.save(model.state_dict(), os.path.join(path, f'SLMC_checkpoint_{epo}.pth'))
        self.best_model = model.state_dict()

    def get_best_model(self, model):
        assert self.best_model is not None
        return model.load_state_dict(self.best_model)
    
class ModelSaver_in_merge:
    def __init__(self, g_logger = None, model_name = "", save_cp = True):
        self.best_acc = -np.Inf
        self.best_model = None
        self.logger = g_logger
        self.model_name = model_name
        self.save_cp = save_cp
    def __call__(self, acc, model, path, t = None):
        if acc > self.best_acc:
            self.save_checkpoint(acc, model, path, t = t)
            self.best_acc = acc
        else:
            self.logger.info(f'No validation improvement - ACC_p: best ({np.abs(self.best_acc):.6f}%) and current ({np.abs(acc):.6f}%). ')
            if t is not None:
                self.logger.info(f'at t={t} ')
    def save_checkpoint(self, acc, model, path, t = None):
        self.logger.info(f'Validation improved (ACC: {np.abs(self.best_acc):.6f} --> {np.abs(acc):.6f}). {"Saving model ..." if self.save_cp else "No cp saving ..."}')
        if t is not None:
                self.logger.info(f'at t={t} ')
        if self.save_cp:
            torch.save(model.state_dict(), os.path.join(path, f'_checkpoint_{self.model_name}.pth'))
            self.best_model = model.state_dict()

    def get_best_model(self, model, path):
        assert self.best_model is not None
        model.load_state_dict(
                torch.load(os.path.join(path, f'_checkpoint_{self.model_name}.pth'), weights_only=True ), 
                strict=True)
        return model