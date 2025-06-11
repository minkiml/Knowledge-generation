import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import time
from tqdm import tqdm
from gradientNNs import TSNets
from .utils import opt_constructor, Value_averager, Logger, grad_logger_spec
from .data_factory.TS_dataloader import Load_dataset
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.collections import LineCollection
import warnings
warnings.simplefilter("ignore", UserWarning)
sns.set(style='ticks', font_scale=1.2)
plt.rcParams['figure.figsize'] = 12,8

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

class EarlyStopping:
    def __init__(self, patience=3, verbose=False,dataset_name='', delta=0, logger = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_MSE = np.Inf
        self.best_MAE = np.Inf

        self.early_stop = False
        self.delta = delta
        self.dataset=dataset_name
        self.logger = logger

    def __call__(self, mse_, mae_, model, path):
        # if mse_ < self.best_MSE:
        # if mae_ < self.best_MAE:
        #     self.save_checkpoint(mse_,mae_, model, path)
        #     self.best_MAE = mae_
        #     self.counter = 0
        if mse_ < self.best_MSE or mae_ < self.best_MAE:
            self.save_checkpoint(mse_,mae_, model, path)
            if mse_ < self.best_MSE:
                self.best_MSE = mse_ 
            if mae_ < self.best_MAE:
                self.best_MAE = mae_
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(f'No validation improvement & EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, mse_, mae_, model, path):
        if self.verbose:
            self.logger.info(f'Validation improved (MSE: {self.best_MSE:.6f} --> {mse_:.6f}) & (MAE: {self.best_MAE} --> {mae_:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, '_checkpoint.pth'))


class Solver(object):
    DEFAULTS = {}
    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        log_loc = os.environ.get("log_loc")
        root_dir = os.getcwd() 
        logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all.txt'), level=logging.INFO,
                            format = '%(asctime)s - %(name)s - %(message)s')
        self.logger = logging.getLogger('Solver')

        data, (_, _) = Load_dataset({"data_path": self.data_path,
                                    "sub_dataset": self.dataset,
                                    "varset_train": self.vars_in_train,
                                    "varset_test": self.vars_in_test,
                                    "channel_dependence": self.channel_dependence,
                                    "training_portion": 0.7,
                                    "look_back": self.look_back,
                                    "horizon": self.horizon,

                                    "batch_training": self.batch,
                                    "batch_testing": self.batch_testing})
        
        self.training_data = data.__get_training_loader__()
        self.testing_data = data.__get_testing_loader__()
        self.val_data = data.__get_val_loader__()
        del data

        self.loss_TD = Value_averager()
        self.loss_FD = Value_averager()
        self.loss_total = Value_averager()

        self.device = torch.device(f'cuda:{self.gpu_dev}' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"GPU (device: {self.device}) used" if torch.cuda.is_available() else 'cpu used')
        self.peak_memory_init = torch.cuda.max_memory_allocated(self.device)
        self.build_model()

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.plots_save_path):
            os.makedirs(self.plots_save_path)
        if not os.path.exists(self.his_save_path):
            os.makedirs(self.his_save_path)
        self.log = Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            ('%.5f', 'loss_TD')
                            )
        
    def build_model(self):
        self.model = {"nlinear": TSNets.NLinear(self.look_back, self.horizon, revin = False)}[self.network]
        ipe = len(self.training_data)
        self.optimizer, self.lr_scheduler, self.wd_scheduler = opt_constructor(self.scheduler,
                                                                            self.model,
                                                                            lr = self.lr_,

                                                                            warm_up = int(self.n_epochs* ipe * self.warm_up),
                                                                            fianl_step = int(self.n_epochs* ipe),
                                                                            start_lr = self.start_lr,
                                                                            ref_lr = self.ref_lr,
                                                                            final_lr = self.final_lr,
                                                                            start_wd = self.start_wd,
                                                                            final_wd = self.final_wd)

        if torch.cuda.is_available():
            self.model.to(self.device)

    def vali(self, dataset_, epo, testing = False):
        self.model.eval()
        all_mse = []
        all_mae = []
        with torch.no_grad():
            for i, (x, y, _) in enumerate(dataset_):
                input = x.to(self.device)######################
                y = y.to(self.device)

                y_horizon_pred = self.model(input)

                if i == 0:
                    all_pred = y_horizon_pred
                    all_y = y
                else:
                    all_pred = torch.cat((all_pred, y_horizon_pred), dim = 0)
                    all_y = torch.cat((all_y, y), dim = 0)

        self.logger.info(f"size: {x.shape}") # To check if downsampled input is correctly made (if applied as in exp section 4.2)

        if (epo % 10) == 0 or testing:
            pass
            # self.log.log_forecasting_vis(y_horizon_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), name_ = f"{epo}" if not testing else "testing")
        if testing:
            pass
            # all_mse = ((all_pred - all_y)**2)[:,-1,:]
            # self.log.log_forecasting_error_vis(all_mse)
        MSE_ = F.mse_loss(all_pred,all_y).detach().cpu().item()
        MAE_ = F.l1_loss(all_pred,all_y).detach().cpu().item()

        return MSE_, MAE_

    def train(self):
        self.logger.info("======================TRAIN MODE======================")

        early_stopping = EarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset, logger=self.logger)
        train_steps = len(self.training_data)
        self.logger.info(f'train_steps: {train_steps}')
        for epoch in tqdm(range(self.n_epochs), desc="Training: "):
            speed_t = []
            epoch_time = time.time()
            self.model.train()
            for i, (x, y, _) in enumerate(self.training_data):
                x = x.to(self.device)######################
                y = y.to(self.device)
                
                if self.lr_scheduler is not None:
                    _new_lr = self.lr_scheduler.step()
                if self.wd_scheduler is not None:
                    _new_wd = self.wd_scheduler.step()
                self.optimizer.zero_grad()
                
                per_itr_time = time.time()

                y_pred = self.model(x)
                loss = F.mse_loss(y_pred, y)

                loss.backward()
                self.optimizer.step()
                speed_t.append(time.time() - per_itr_time)
                
                self.loss_TD.update(loss.item())
                self.loss_total.update(loss.item())
                self.log.log_into_csv_(epoch+1,
                                            i,
                                            self.loss_TD.avg
                                            )
                if (i + 1) % 100 == 0:
                    self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}] & s/iter:{np.mean(speed_t): .5f}, left time: {np.mean(speed_t) * (train_steps - i): .5f}, Loss_TD:{self.loss_TD.avg: .4f}")
                

            self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}] & speed per epoch: {(time.time() - epoch_time): .5f}")
            
            MSE_, MAE_ = self.vali(self.val_data, epoch+1)
            MSE_t, MAE_t = self.vali(self.testing_data, 1)
            self.logger.info(f"Epoch[{epoch+1}],  MSE: {MSE_: .5f} & MAE: {MAE_: .5f}")
            self.logger.info(f"Epoch[{epoch+1}], TESTING --  MSE: {MSE_t: .5f} & MAE: {MAE_t: .5f}")
            early_stopping(MSE_, MAE_,  self.model, self.model_save_path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
            self.logger.info("")

    def test(self):
        if os.path.exists(os.path.join(str(self.model_save_path), '_checkpoint.pth')):
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), '_checkpoint.pth'),weights_only=True))
            self.logger.info("Best trained Model called:")
        else: 
            raise ImportError(self.logger.info("Loading checkpoint model failed"))
        self.model.eval()
        self.logger.info("======================TEST MODE======================")
        MSE_, MAE_ = self.vali(self.testing_data, 0, testing= True)

        self.logger.info(f"Forecasting result - MSE: {MSE_} & MAE: {MAE_}")
    

    def analysis(self, num = 5):
        if os.path.exists(os.path.join(str(self.model_save_path), '_checkpoint.pth')):
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), '_checkpoint.pth'),weights_only=True))
            self.logger.info("Best trained Model called:")
        else: 
            raise ImportError(self.logger.info("Loading checkpoint model failed"))
        self.model.eval()
        self.logger.info("======================ANALYSIS MODE======================")
        gradients = []
        activations = []

        # Register hook on the last conv layer
        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        for j in range(num):
            for i, (x,y,_) in enumerate([next(iter(self.training_data)), next(iter(self.testing_data))]):
                input = x.to(self.device).detach().requires_grad_()######################
                y = y.to(self.device)
                
                y_pred = self.model(input[0:1]) # (1,L, c)
                score = y_pred[0,:,0].sum() # channel 0 (does not matter since channel independence is made)
                self.model.zero_grad()
                score.backward()
                
                prediction_error = F.mse_loss(y_pred, y, reduction='none').detach().cpu()[0,:,0]
                grad_input = input.grad.detach().cpu()[0,:,0]
                # print(grad_input)
                grad_weights = get_weight_grad(self.model, 0)
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)

                # --- Subplot 1: Heatmap of grad_weights ---
                im0 = axes[0].imshow(grad_weights.T, cmap='bwr', aspect='auto')
                axes[0].set_title("Gradient of Weights")
                axes[0].set_ylabel("in_dim")
                axes[0].set_xlabel("out_dim")
                fig.colorbar(im0, ax=axes[0])

                # --- Subplot 2: Line plot of grad_input ---
                axes[1].plot(grad_input, color='blue')
                axes[1].set_title("Gradient of Input")
                axes[1].set_ylabel("Amplitude")
                axes[1].set_xlabel("Length")
                
                
                # input_seq = input.detach().cpu()[0,:,0]
                # output_seq = y_pred.detach().cpu()[0,:,0]
                # L = len(input_seq)
                # H = len(output_seq)
                # x_input = np.arange(L)
                # y_input = input_seq

                # # Make segments between points (x0,y0) to (x1,y1)
                # points = np.array([x_input, y_input]).T.reshape(-1, 1, 2)
                # segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # # Normalize grad_input for color mapping
                # norm = plt.Normalize(grad_input.min(), grad_input.max())
                # line_collection = LineCollection(segments, cmap='viridis', norm=norm)
                # line_collection.set_array(grad_input)
                # line_collection.set_linewidth(2)

                # # Add colored line to subplot
                # axes[2].add_collection(line_collection)
                # axes[2].set_xlim(x_input.min(), x_input.max())
                # axes[2].set_ylim(y_input.min(), y_input.max())

                # # Plot output_seq as normal (after input ends)
                # x_output = np.arange(L, L + H)
                # axes[2].plot(x_output, output_seq, color='green', label='Output Seq')

                # # Axis labels, title, and legend
                # axes[2].set_title("Input → Output Sequence with Gradient Coloring")
                # axes[2].set_ylabel("Amplitude")
                # axes[2].set_xlabel("Time Step")
                # axes[2].legend(loc='upper right')

                # # Add colorbar for gradient magnitude
                # cbar = fig.colorbar(line_collection, ax=axes[2])
                # cbar.set_label("Gradient w.r.t. Input")
              
                input_seq = input.detach().cpu()[0,:,0]
                output_seq = y_pred.detach().cpu()[0,:,0]
                x_input = range(len(input_seq))
                x_output = range(len(input_seq), len(input_seq) + len(output_seq))
                
                # --- Subplot 3: original input sequence line plot ---
                axes[2].plot(x_input, input_seq, color='red', label='Input Seq')
                axes[2].plot(x_output, output_seq, color='green', label='Prediction')
                axes[2].set_title("Input → Output Sequence")
                axes[2].set_ylabel("Amplitude")
                axes[2].set_xlabel("Time Step")
                axes[2].legend(loc='upper right')


                # --- Save figure ---
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_save_path, f"{j}_gradients_weights_inputs_{i}.png"))
                plt.clf()
                plt.close(fig)
            # fig = plt.figure(figsize=(12, 8), 
            #         dpi = 600) 
            # axes = fig.subplots()
            # im = plt.imshow(grad_weights.T, cmap='bwr', aspect='auto')
            # plt.ylabel("in_dim")
            # plt.xlabel("out_dim")
            # # Add a colorbar using the ScalarMappable
            # cbar = plt.colorbar(im)
            # plt.savefig(os.path.join(self.plots_save_path, f"gradients_weights_{i}.png")) 
            # plt.clf()   
            # plt.close(fig)
            
            # fig = plt.figure(figsize=(12, 8), 
            #         dpi = 600) 
            # axes = fig.subplots()
            # im = plt.plot(grad_input, color = "blue") # , cmap='bwr'
            # plt.ylabel("Amplitude")
            # plt.xlabel("length")
            # # Add a colorbar using the ScalarMappable
            # # cbar = plt.colorbar(im)
            # plt.savefig(os.path.join(self.plots_save_path, f"gradients_inputs_{i}.png")) 
            # plt.clf()   
            # plt.close(fig)
             
def get_weight_grad(model: nn.Module, layer_index: int):
    nn_to_be_layers = (nn.Conv2d, nn.Linear)
    
    target_layer = [m for m in model.modules() if isinstance(m, nn_to_be_layers)]

    if layer_index < 0 or layer_index >= len(target_layer):
        raise ValueError(f"Layer index {layer_index} is out of bounds. Found {len(target_layer)} eligible layers.")

    layer = target_layer[layer_index]

    if layer.weight.grad is None:
        return None

    return layer.weight.grad.detach().cpu()