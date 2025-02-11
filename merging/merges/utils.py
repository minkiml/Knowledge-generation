import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os

from torch.utils.data import DataLoader, ConcatDataset, Subset

def load_model(model, model_path, model_name, loc_ = None, g_logger = None):
    '''loc_ needs to be the full directory to .pth 
    if not speficied model is tried to be drawn from defalu path (model path)'''
    try:
        model.load_state_dict(
                            torch.load( loc_ if loc_ is not None else os.path.join(str(model_path), f'_checkpoint_{model_name}.pth') ))
        g_logger.info("Pretrained model has been successfully loded")
        return model, False
    except:    
        g_logger.info("No checkpoint exists in the designated location.")
        g_logger.info("Returning fresh model")
        return model, True
    
def save_model(model, model_path, model_name):
    model.eval()
    torch.save(model.state_dict(), os.path.join(model_path, f'_checkpoint_{model_name}.pth'))


def get_module(m, module_name):
    '''Get module (module_name) from model m'''
    current_module = m
    for name in module_name.split('.'):
        current_module = getattr(current_module, name)
    return current_module

def update_module(model, full_name, new_layer):
    ''' Update the module (full name) with a new moduel (new_layer)  in the model'''
    parts = full_name.split('.')
    target = model
    for part in parts[:-1]:  
        target = getattr(target, part)
    setattr(target, parts[-1], new_layer)

def merge_dataloader(dataloader_A: DataLoader, # conventional dataloader object
                     dataloader_B: DataLoader,
                     
                     batch_size = 256,
                     subset_p = 1. # e.g., (smaller) 40 % from each dataset is combined  
                     ):
    ''' Combine dataloaders into single mixture dataloader '''
    if subset_p < 1.:
        # TODO: add different strategies to mix and pick the portion from each dataset. 
        subset_num_A = int(len(dataloader_A.dataset) * subset_p)
        subset_num_B = int(len(dataloader_B.dataset) * subset_p)
        subset_num = subset_num_A if subset_num_A < subset_num_B else subset_num_B

        subset_A = Subset(dataloader_A.dataset, random.sample(range(len(dataloader_A.dataset)), subset_num))
        subset_B = Subset(dataloader_B.dataset, random.sample(range(len(dataloader_B.dataset)), subset_num))
    else:
        # Total sets
        subset_A = dataloader_A.dataset
        subset_B = dataloader_B.dataset
    return DataLoader(ConcatDataset([subset_A, subset_B]), batch_size=batch_size, shuffle=True,
                      drop_last= True)

def merge_data_iter(*data_loader):
    """Put multiple dataloaders into a tuple of iterables."""
    if len(data_loader) == 1 and not isinstance(data_loader[0], (tuple, list)):
        return (data_loader[0],)  # Single data loader, wrap it in a tuple
    return tuple(data_loader)  # Convert multiple arguments into a tuple

def equalize_init(ref, target):
    '''Equalize the target network's initialization to reference network'''
    with torch.no_grad():
        for param, param_B in zip(ref.parameters(), target.parameters()):
            param_B.copy_(param.data.detach().clone())
    return ref, target

def reset_batchnorm_stat(f):
    '''Reset the statistics of batchnorms in the network f'''
    with torch.no_grad():
        for name, module in f.named_modules():
            if isinstance(module, (BatchNorm2d, nn.BatchNorm2d)):
                module.running_mean.zero_()  
                module.running_var.fill_(1) 
    return f

def print_with_linespace(net_profile, global_print= None):
    #print dictionary with line space
    output = ""
    for key in net_profile:
        output += f"'{key}': {net_profile[key]},\n"
    formatted_output = output.replace("},", "',\n")
    print(formatted_output) if global_print is None else global_print.info(formatted_output)




class SkipConnection(nn.Module):
    '''
    Explicit skip connection with forward identity.
    '''
    def __init__(self):
        super(SkipConnection, self).__init__()
        self.res= nn.Identity()
    def forward(self, x):
        return self.res(x)


# class SkipConnection_P(nn.Module): # TODO: This requires modification if nested merging is required.
#     '''
#     Explicit skip connection with forward identity, playing as a placeholder for easy merging.

#     Note that having inverse and forward transforms for empty skip connection
#     assigns a new dense matrix to the originally empty skip connection.   
#     '''
#     def __init__(self):
#         super(SkipConnection_P, self).__init__()

#         self.register_parameter("weight", None)
#         self.register_parameter("bias", None)

#     def forward(self, x):
#         # if self.param:
#         if x.dim() == 4:
#             # 2d Conv case
#             x = F.linear(x.permute(0,2,3,1), self.weight, self.bias).permute(0,3,1,2)
#         else:
#             # (..., C) case
#             x = F.linear(x, self.weight, self.bias)
#         # else: raise NotImplementedError("")
#         return x

class SkipConnection_P(nn.Linear): # TODO: This requires modification if nested merging is required.
    '''
    Explicit skip connection with forward identity, playing as a placeholder for easy merging.

    Note that having inverse and forward transforms for empty skip connection
    assigns a new dense matrix to the originally empty skip connection.   
    '''
    def __init__(self, in_features, out_features, bias=False):
        super(SkipConnection_P, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        # Initialize weight as an identity matrix
        nn.init.eye_(self.weight)

        # Initialize the bias (default behavior)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # if self.param:
        if input.dim() == 4:
            # 2d Conv case
            input = input.permute(0,2,3,1)
            output = super().forward(input).permute(0,3,1,2)
        else:
            # (..., C) case
            output = super().forward(input)
        return output

class LayerNorm(nn.Module):
    '''
    Note that n-dimensional scaling weight is n-dimensional diagonal matrix.
    After composition, this becomes a full dense matrix.
    '''
    def __init__(self, hidden = 64, initial_affine = False, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.affine = True
        self.eps = eps
        if initial_affine:
            weight = nn.Parameter(torch.zeros((hidden, hidden))) # (outdim, indim) or (indim,) equiv. diagonal(indim)
            bias = nn.Parameter(torch.zeros(hidden))
            self.register_parameter("weight", weight) 
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize the input
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_normalized = F.linear(x_normalized, self.weight, self.bias)
        return x_normalized

class BatchNorm2d(nn.Module):
    '''
    Note that n-dimensional (channel dimension) scaling weight is n-dimensional diagonal matrix.
    After composition, this becomes a full dense matrix.

    For merging, we only modify affine parameters but not statistics for which we "re-estimate" 
    after reset (1) during inference or (2) after mering. 
    
    No affine params treated as identity affine during merging unless this initially-unlearnable normalization layer was opted out for merging.
    '''
    def __init__(self, hidden = 64, initial_affine = False, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.hidden = hidden
        self.eps = eps
        self.momentum = momentum
        self.affine = True

        if initial_affine:
            self.weight = nn.Parameter(torch.ones(hidden))
            self.bias = nn.Parameter(torch.zeros(hidden))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(hidden))
        self.register_buffer('running_var', torch.ones(hidden))

    def forward(self, x):
        if self.training:
            mean = x.mean([0, 2, 3], keepdim=True)  # Mean across batch and spatial dimensions
            var = x.var([0, 2, 3], keepdim=True, unbiased=False)  # Variance across batch and spatial dimensions

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.view(1, self.hidden, 1, 1)
            var = self.running_var.view(1, self.hidden, 1, 1)
        # Normalize the input
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            # x_normalized = torch.einsum('...i, ij -> ...j', 
            #                             x_normalized.permute(0,2,3,1), 
            #                             self.weight.T).permute(0,3,1,2) + self.bias.view(1, -1, 1, 1)
            
            x_normalized = F.linear(x_normalized.permute(0,2,3,1), self.weight, self.bias).permute(0,3,1,2)
        return x_normalized