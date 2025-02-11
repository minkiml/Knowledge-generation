import torch
import torch.nn.functional as F
import numpy as np
import copy

from merges.utils import BatchNorm2d
from torch import nn
from merges import utils
def vali(f, data_loader_test, device):
        f.train(False)
        with torch.no_grad():
            for i, batches in enumerate(zip(*data_loader_test)):
                if len(batches) == 1:
                    x, y = batches[0]
                else:
                    x = torch.cat((batches[0][0],batches[1][0]), dim = 0)
                    y = torch.cat((batches[0][1],batches[1][1]), dim = 0)
                input = x.to(device)######################
                y = y.to(device)
                logit = f(input)
                pred = F.log_softmax(logit, dim = -1).argmax(-1)
                ###########################
                if i == 0:
                        all_pred = pred
                        all_y = y
                else:
                    all_pred = torch.cat((all_pred, pred), dim = 0)
                    all_y = torch.cat((all_y, y), dim = 0)
                     
        all_pred = all_pred.detach().cpu().numpy()
        all_y = all_y.detach().cpu().numpy()
        num_ = all_pred.shape[0]

        ACC_ = (all_pred == all_y).sum() / num_
        ACC_ *= 100
        return ACC_


def recompute_batchnorm_stat(f, testing_data, device, repeat = 1, reset = True):
    if reset:
        utils.reset_batchnorm_stat(f)
    f.train()
    with torch.no_grad():
        for _ in range(repeat):
            for i, batches in enumerate(zip(*testing_data)):
                if len(batches) == 1:
                    x, y = batches[0]
                else:
                    x = torch.cat((batches[0][0],batches[1][0]), dim = 0)
                    y = torch.cat((batches[0][1],batches[1][1]), dim = 0)
                input = x.to(device)######################
                y = y.to(device)
                _ = f(input)
                # pred = F.log_softmax(logit, dim = -1).argmax(-1)
    f.eval()
    return f



def averaging_weight(m1, m2):
    # TODO: implement handling norms

    with torch.no_grad():
        model_avg = copy.deepcopy(m1)

        # Average the parameters
        for param_A, param_B, param_avg in zip(m1.parameters(), m2.parameters(), model_avg.parameters()):
            param_avg.data = (param_A.data + param_B.data) / 2

        return model_avg
    

def inter_1d(model_a, model_b, testing_data, 
             num_alpha = 20, device = None):
    model_a.to(device)
    model_b.to(device)
    # print(model_a)
    # print(model_b)
    # raise NotImplementedError()
    avg_model = copy.deepcopy(model_a)
    alpha_list = torch.linspace(0, 1, steps=num_alpha)
    acc_list = []
    for i, alpha in enumerate(alpha_list):
        if i == 0:
            pass
        elif i == (num_alpha - 1):
            avg_model = model_b
         
        else:

            state_dict_a = model_a.state_dict()
            state_dict_b = model_b.state_dict()
            averaged_state_dict = {}
            for key in state_dict_a:
                averaged_state_dict[key] = (alpha * state_dict_b[key]) + ((1 - alpha) * state_dict_a[key])
            avg_model.load_state_dict(averaged_state_dict)

        acc = vali(avg_model.to(device), testing_data= testing_data, device= device)

        acc_list.append(acc)


    return acc_list, alpha_list

    