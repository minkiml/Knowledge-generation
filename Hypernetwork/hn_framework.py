import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy
import torch
import os
from .hypernets import *
from .compression import *

def MSE_loss(target, pred):
    return torch.mean((target-pred)**2)

def MAE_loss(target, pred):
    return torch.mean(target.abs()-pred.abs())

class Framework_HN(nn.Module):
    '''
    ...
    node_direction: specify which direction of nodes (input nodes or output nodes) of layers to learn. It determines
                    how the hypernetwork generates parameters for the implicit target network.  
    '''
    def __init__(self, network, dir = None, name = "", device = "cpu",
                 init_net = None, 
                 compression = False, 
                 hypernet_base = "mlp",
                 node_direction = "W",
                 lowrank = False,
                 rank = 4,
                 type_ = "lowrank",
                 decomposition = False,
                 learnable_emb = False,
                 hypermatching = False,
                 zero_init_emb = False,
                 intrinsic_training = False):
        super(Framework_HN, self).__init__()
        self.device = device
        self.dir = dir
        self.compression = compression 
        self.node_direction = node_direction 
        self.lowrank = lowrank
        self.decomposition = decomposition
        self.intrinsic_training = intrinsic_training
        self.learnable_emb = learnable_emb if not intrinsic_training else True
        self.hypermatching = hypermatching if not intrinsic_training else False
        self.zero_init_emb = zero_init_emb
        
        if self.decomposition:
            self.compression = True
            
        self.rank = rank
        self.type_ = type_
        if self.lowrank:
            self.node_direction = "W"
        
        self.init_Implicitnet(network, init_net)
        self.init_Hypernet(type = hypernet_base)
        
        if dir is not None:
            self.load_hypernet(name)
    def forward(self, x):
        # Training forward?
        # generation -> reconstruction -> yield loss ? 
        pass
    
    def forward_implicitnet(self, x):
        self.forward_Hypernet("updating")
        y = self.implicitnet[0](x)
        return y
    
    def forward_Hypernet(self, mode = "updating", emb_in = None):
        # Generation of parameters (forward of hypernet) for the implicit net
        
        index = 0
        # Iterate over the layers
        losses = []
        errors = torch.tensor(0.).to(self.device)
        
        if self.lowrank:
            # forward processing of embeddings for all layers
            self.hypernet.forward_emb(emb_in = emb_in)
        
        for i, (name, base, localname) in enumerate(self.name_base_localname):
            if mode == "updating":
                # Update the implicit net
                params = self.hypernet.generate_W(name, decom = False, emb_dict = emb_in)
                if self.init_params is not None:
                    params = self.init_params[name] + params
                setattr(base, localname, params)
                
            elif mode == "learning":
                params = self.hypernet.generate_W(name, decom = self.decomposition, emb_dict = emb_in)
                if isinstance(params, list):
                    # U, V, S = params[0], params[1], params[2]
                    
                    # Compute error (layer wise)
                    for j, (pred, target) in enumerate(zip(params, [self.target_params[name]["U"], self.target_params[name]["V"], self.target_params[name]["S"] ])):
                        if j == 0:
                            error = torch.mean(torch.abs(pred - target))#((target - pred)**2).mean()
                        else:
                            error += torch.mean(torch.abs(pred - target)) # ((target - pred)**2).mean()
                    errors += error # / 3.
                else:
                    if (self.init_params is not None) and not (self.compression) and not (self.decomposition):
                        params = self.init_params[name] + params
                    
                    errors +=  torch.mean(torch.abs(params - self.target_params[name])) #torch.mean(torch.abs(params - self.target_params[name])) #((self.target_params[name] - params)**2).mean()
                        
                    # Compute error (layer wise)
                    
                # Hypernetwork learning by reconstructing
                
                # Layer wise
                # MSE
                # loss = MSE_loss(target = self.target_params[name], 
                #                 pred = params)
                
                # As a whole
                # errors.append(torch.flatten((self.target_params[name] - params)**2))
                
        if mode == "learning":
            return errors #/ (i+1)#torch.concat((errors), dim  = 0).mean()
        else: 
            return None 
    
    def init_Hypernet(self, hypernet_fr = None, type = "mlp"):
        '''Init hypernetwork for the implicit target network'''
        if hypernet_fr is None:
            self.hypernet = {"mlp": Hypernet_MLP(self.frame, hidden_dim=512, 
                                                 iteration = 10, node_direction= self.node_direction, 
                                                 lowrank = self.lowrank, rank = self.rank, 
                                                 type_ = self.type_, learnable_emb=self.learnable_emb, 
                                                 zero_init_emb = self.zero_init_emb,
                                                 device = self.device),
                             "transformer": Hypernet_TRF(self.frame, hidden_dim=512, 
                                                         iteration = 40, num_layer = 3, 
                                                         node_direction=self.node_direction, lowrank = self.lowrank, 
                                                         rank = self.rank, type_ = self.type_, 
                                                         learnable_emb=self.learnable_emb, 
                                                         zero_init_emb = self.zero_init_emb, device = self.device, masking= True),
                             "lstm": Hypernet_LSTM(self.frame, hidden_dim=512, 
                                                    iteration = 1, num_layer = 3, 
                                                    node_direction=self.node_direction, lowrank = self.lowrank, 
                                                    rank = self.rank, type_ = self.type_, 
                                                    learnable_emb=self.learnable_emb, zero_init_emb = self.zero_init_emb, 
                                                    device = self.device),
                             "mlpmixer": Hypernet_mixer(self.frame, hidden_dim=512, 
                                                         iteration = 1, num_layer = 3, 
                                                         node_direction=self.node_direction, lowrank = self.lowrank, 
                                                         rank = self.rank, type_ = self.type_, 
                                                         learnable_emb=self.learnable_emb, zero_init_emb = self.zero_init_emb,
                                                         device = self.device)}[type]
            self.hypernet.to(self.device)
        else:
            with torch.no_grad():
                #deep copy such that it has the same initialization as the passed arg
                if not self.hypermatching:
                    self.hypernet = copy.deepcopy(hypernet_fr.hypernet)
                else: self.hypernet = hypernet_fr.hypernet # SHARE
                self.hypernet.to(self.device)
            
    def init_Implicitnet(self, network, init_net = None): # TODO when implementing it, need to consider how to design hypernetwork 
        '''Assign pretrained target network'''

        network.eval()
        implicitnet = copy.deepcopy(network)
        self.implicitnet = [implicitnet] # this is wrapped in python list, thus ruleing out of the nn calls (e.g., .train)
        self.name_base_localname = []

        if self.compression:
            self.target_params = dict() 
            self.compressor = LayerCompressorSVD(k= self.rank)
        else:
            self.target_params = dict() # Ground truth
            self.compressor = None
        
        if init_net is not None:
            self.init_params = dict()
        else: self.init_params = None
        self.frame = dict()
        # Iterates over layers in the Neural Network
        length = 0
        for name, param in implicitnet.named_parameters():
            # If the parameter requires gradient update
            if param.requires_grad:
                length += 1
                param = param.clone().detach().requires_grad_(False).to(self.device)
                shape_param = param.shape
                
                if (self.compression) and (param.dim() != 1) and (self.decomposition):
                    decom_params = self.compressor.compress(param)
                    self.target_params[name] = {"U":decom_params[0],
                                                "V":decom_params[1],
                                                "S":decom_params[2]
                                                }
                else:
                    self.target_params[name] = (param)

                
                if (self.init_params is not None) and not (self.compression) and not (self.decomposition):
                    init_value = dict(init_net.named_parameters())[name]
                    self.init_params[name] = (
                        init_value.clone().detach().requires_grad_(False).to(self.device)
                    )
                
                base, localname = implicitnet, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))
              
                self.frame[name] = {"shape": shape_param, #permute_shape(param.shape, param.dim()) if self.node_direction == "H" else param.shape,# param.shape, ## HERE TODO PERMUTE
                                    "module": base.__class__.__name__,
                                    "local_name": localname # TODO how to handle other modules like Embedding, nn.parameters, etc.
                                    }
        # Delete the network's param attributes after extracting the info
        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

    ######################
    ######################
    def vis_embeddings_out(self, vis_function, etc = ""):
        with torch.no_grad():
            self.implicitnet_train(False)
            for name, base, localname in self.name_base_localname:
                if self.lowrank:
                    # forward processing of embeddings for all layers
                    self.hypernet.forward_emb()
                emb_out = self.hypernet.generate_W(name, outemb = True, emb_dict = None)
                if emb_out is not None:
                    # dictionary to tensor
                    emb_out = torch.cat([v for v in emb_out.values()], dim=0)
                    vis_function(emb_out, f"outemb_{etc}_{name}")
    def vis_embeddings(self, vis_function):
        if self.hypernet.fixed_embeddings is not None:
            for i in range (1, self.hypernet.depth + 1):
                emb = self.hypernet.get_emb(i)
                vis_function(emb, f"layer_{i}")
        else:
            vis_function(self.hypernet.all_emb, f"layers")       
            
    def assign_hypernet_emb(self, hypernet_fr):
        for i in range (1, self.hypernet.depth + 1):
            emb = self.hypernet.get_emb(i)
            to_emb = hypernet_fr.hypernet.get_emb(i)
            emb.copy_(to_emb)
            
    def get_Hypernet(self):
        with torch.no_grad():
            self.hypernet.eval()
            return copy.deepcopy(self.hypernet)
        
    def get_subspace_emb(self, learnable = False):
        with torch.no_grad():
            if isinstance(self.hypernet.all_emb, dict):
                return copy.deepcopy(self.hypernet.emb_dict) if not learnable \
                    else {k: nn.Parameter(v.detach().clone()) for k, v in self.hypernet.emb_dict.items()}
            else:
                return self.hypernet.all_emb.detach().clone() if not learnable \
                    else nn.Parameter(self.hypernet.all_emb.detach().clone())
    
    def set_subspace_emb(self, subspace_emb):
        with torch.no_grad():
            if isinstance(subspace_emb, dict):
                if self.learnable_emb:
                    self.hypernet.emb_dict = {k: nn.Parameter(v.detach().clone()) for k, v in subspace_emb.items()}
                else:
                    self.hypernet.emb_dict = copy.deepcopy(subspace_emb)
            else:
                if self.learnable_emb:
                    self.hypernet.all_emb = nn.Parameter(subspace_emb.detach().clone())
                else:
                    self.hypernet.all_emb = subspace_emb.detach().clone()

    def marterialize_Implicitnet(self, m, emb_in = None):# TODO 
        with torch.no_grad():
            self.implicitnet_train(False)
            self.hypernet.eval()
            m.eval()
            # update the mainnet based on the so-far trained hypernet then instantiate out
            if self.lowrank:
                # forward processing of embeddings for all layers
                self.hypernet.forward_emb(emb_in=emb_in)
            for name, param in m.named_parameters():
                gen_params = self.hypernet.generate_W(name, emb_dict = emb_in)
                if self.init_params is not None:
                    gen_params = self.init_params[name] + gen_params
                param.data.copy_(gen_params)
            # self.forward_Hypernet("updating")
            return m
            
    def implicitnet_train(self, train = True):
        if train:
            self.implicitnet[0].train()
        else:
            self.implicitnet[0].eval()    
    def save_hypernet(self, name):
        self.hypernet.eval()
        torch.save(self.hypernet.state_dict(), os.path.join(self.dir, f'_checkpoint_{name}.pth'))
        
    def load_hypernet(self,name):
        try:
            self.hypernet.load_state_dict(
                                torch.load( os.path.join(str(self.dir), f'_checkpoint_{name}.pth'), 
                                        weights_only=True ), strict=True)
            print("Pretrained hypernet has been successfully loded")
            self.checkpoint = True
        except:    
            print("No hypernet checkpoint exists in the designated location.")
            self.checkpoint = False
