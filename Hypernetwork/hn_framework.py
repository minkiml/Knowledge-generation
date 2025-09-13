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
                 decomposition = False, # Directly reconstruct compressed layer matrices (e.g., low rank with svd)  
                 learnable_emb = False, # Set the input embeddings to be learnable otherwise fixed
                 hypermatching = False, # Share the hypernetwork across all candidate main nets
                 zero_init_emb = False, # Init the input embeddings to zero
                 intrinsic_training = False, # Leanring like learning intrinsic dimension (training embeddings only) 
                 hyper_grad = False, # Whether to use hypernet with grad training
                 grad_learning = False,
                 max_T = None
                 ):
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
        self.hyper_grad = hyper_grad
        self.grad_learning = grad_learning
        self.max_T = max_T
        
        if self.decomposition:
            self.compression = True
            
        self.rank = rank
        self.type_ = type_
        if self.lowrank:
            self.node_direction = "W"
        
        self.init_Implicitnet(network, init_net
                              )
        self.init_Hypernet(type = hypernet_base)
        
        if dir is not None:
            self.load_hypernet(name)
        
        if grad_learning:
            self.set_copy_initparam()
    def forward(self, x):
        # Training forward?
        # generation -> reconstruction -> yield loss ? 
        pass
    
    def forward_implicitnet(self, x, t = None, isolate_hypernet = False, params = None, 
                            grad_learning = False): # TODO add conditioinal t
        if not grad_learning:
            self.forward_Hypernet("updating", t= t, isolate_hypernet = isolate_hypernet, params = params)
        else:
            # set the main net with the non-learnable init params
            self.forward_with_initparam()
        y = self.implicitnet[0](x)# for grad --> how to get grad from these
        return y
    
    def forward_with_initparam(self):
        for i, (name, base, localname) in enumerate(self.name_base_localname):
            params = self.init_params[name]
            params = params.detach().clone().requires_grad_(True)
            setattr(base, localname, params)
            
    def forward_Hypernet(self, mode = "updating", 
                         emb_in = None, t = None, 
                         isolate_hypernet = False, 
                         grad_learning = False,
                         params = None): # TODO add conditioinal t
        # Generation of parameters (forward of hypernet) for the implicit net
        
        index = 0
        # Iterate over the layers
        losses = []
        gen_prams = []
        errors = torch.tensor(0.).to(self.device)
        
        if params is not None:
            for i, (name, base, localname) in enumerate(self.name_base_localname):
                if isolate_hypernet:
                    param = params[i].detach().requires_grad_(True)
                else: param = params[i].detach()
                setattr(base, localname, param)
                
        else:
            
            if self.lowrank:
                # forward processing of embeddings for all layers
                self.hypernet.forward_emb(emb_in = emb_in, t = t)

            for i, (name, base, localname) in enumerate(self.name_base_localname):
                if mode == "updating":
                    # Update the implicit net
                    params = self.hypernet.generate_W(name, decom = False, emb_dict = emb_in) # TODO incorporate T
                    
                    if (self.hyper_grad):
                        params *= t
                        
                    if self.init_params is not None:
                        params = self.init_params[name] + params
                    
                    if isolate_hypernet:
                        params = params.detach().clone().requires_grad_(True)
                        
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
                elif mode == "generating": 
                    params = self.hypernet.generate_W(name, decom = False, emb_dict = emb_in)
                    if grad_learning:
                        pass
                    else:
                        if (self.hyper_grad):
                            params *= t
                        if isolate_hypernet:
                                params = params.detach().clone().requires_grad_(True)
                                
                        if (self.init_params is not None) and not (self.compression) and not (self.decomposition):
                            params = self.init_params[name] + params
                    gen_prams.append(params)
                    
                elif mode == "hyperout": 
                    params = self.hypernet.generate_W(name, decom = False, emb_dict = emb_in)
                    
                    if (self.hyper_grad):
                        params *= t
                    if isolate_hypernet:
                            params = params.detach().clone().requires_grad_(True)
                    gen_prams.append(params)
                    # Hypernetwork learning by reconstructing
                    
                    # Layer wise
                    # MSE
                    # loss = MSE_loss(target = self.target_params[name], 
                    #                 pred = params)
                    
                    # As a whole
                    # errors.append(torch.flatten((self.target_params[name] - params)**2))
                    
            if mode == "learning":
                return errors #/ (i+1)#torch.concat((errors), dim  = 0).mean()
            elif (mode == "generating") or (mode == "hyperout"):
                return gen_prams 
            else: 
                return None 
    def reset_initparams(self, newset = None):
        if newset is None:
            for i, (name, base, localname) in enumerate(self.name_base_localname):
                params = getattr(base, localname)
                self.init_params[name] = params.clone().detach().requires_grad_(False).to(self.device)
        else:
            for i, (name, base, localname) in enumerate(self.name_base_localname):
                if isinstance(newset, dict): 
                    self.init_params[name] = newset[name].clone().detach().requires_grad_(False).to(self.device)
                else:
                    self.init_params[name] = newset[i].clone().detach().requires_grad_(False).to(self.device)

    
    def init_Hypernet(self, hypernet_fr = None, type = "mlp"):
        '''Init hypernetwork for the implicit target network'''
        if hypernet_fr is None:
            hypernet = {"mlp": Hypernet_MLP,
                        # (self.frame, hidden_dim=1024, 
                        #                          iteration = 4, num_layer = 2, node_direction= self.node_direction, 
                        #                          lowrank = self.lowrank, rank = self.rank, 
                        #                          type_ = self.type_, learnable_emb=self.learnable_emb, 
                        #                          zero_init_emb = self.zero_init_emb, cond_dim = 32, cond_emb_type = "linear",
                        #                          hyper_grad = self.hyper_grad,
                        #                          device = self.device),
                             "transformer": Hypernet_TRF,
                            #  (self.frame, hidden_dim=256, 
                            #                              iteration = 2, num_layer = 3, 
                            #                              node_direction=self.node_direction, lowrank = self.lowrank, 
                            #                              rank = self.rank, type_ = self.type_, 
                            #                              learnable_emb=self.learnable_emb, 
                            #                              zero_init_emb = self.zero_init_emb, cond_dim = 32, cond_emb_type = "linear",
                            #                              hyper_grad = self.hyper_grad,
                            #                              device = self.device, masking= True),
                             "lstm": Hypernet_LSTM,
                            #  (self.frame, hidden_dim=512, 
                            #                         iteration = 1, num_layer = 3, 
                            #                         node_direction=self.node_direction, lowrank = self.lowrank, 
                            #                         rank = self.rank, type_ = self.type_, 
                            #                         learnable_emb=self.learnable_emb, zero_init_emb = self.zero_init_emb, 
                            #                         cond_dim = 32, cond_emb_type = "linear", hyper_grad = self.hyper_grad,
                            #                         device = self.device),
                             "mlpmixer": Hypernet_mixer,
                            #  (self.frame, hidden_dim=512, 
                            #                              iteration = 1, num_layer = 3, 
                            #                              node_direction=self.node_direction, lowrank = self.lowrank, 
                            #                              rank = self.rank, type_ = self.type_, 
                            #                              learnable_emb=self.learnable_emb, zero_init_emb = self.zero_init_emb,
                            #                              cond_dim = 32, cond_emb_type = "linear", hyper_grad = self.hyper_grad,
                            #                              device = self.device),
                             "identity": Hypernet_Identity
                            #  (self.frame, hidden_dim=1024, 
                            #                      iteration = 4, node_direction= self.node_direction, 
                            #                      lowrank = self.lowrank, rank = self.rank, 
                            #                      type_ = self.type_, learnable_emb=self.learnable_emb, 
                            #                      zero_init_emb = self.zero_init_emb,
                            #                      cond_dim = 256, cond_emb_type = "linear", hyper_grad = self.hyper_grad,
                            #                      device = self.device)
                             }[type]
            self.hypernet = hypernet(self.frame, hidden_dim=256, 
                                    iteration = 2, num_layer = 3, 
                                    node_direction=self.node_direction, lowrank = self.lowrank, 
                                    rank = self.rank, type_ = self.type_, 
                                    learnable_emb=self.learnable_emb, 
                                    zero_init_emb = self.zero_init_emb, cond_dim = 32, cond_emb_type = "linear",
                                    hyper_grad = self.hyper_grad, base = type,
                                    device = self.device, masking= True)
            self.hypernet.to(self.device)
        else:
            with torch.no_grad():
                #deep copy such that it has the same initialization as the passed arg
                if not self.hypermatching:
                    print("no shared net")
                    self.hypernet = copy.deepcopy(hypernet_fr.hypernet)
                else: 
                    self.hypernet = hypernet_fr.hypernet # SHARE
                    print("shared net init")
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
    def set_copy_initparam(self):
        self.init_param_0 = copy.deepcopy(self.init_params)
    ######################
    ######################
    def vis_embeddings_out(self, vis_function, etc = "", t=None):
        with torch.no_grad():
            self.implicitnet_train(False)
            for name, base, localname in self.name_base_localname:
                if self.lowrank:
                    # forward processing of embeddings for all layers
                    self.hypernet.forward_emb(t=t)
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

    def marterialize_Implicitnet(self, m, emb_in = None, t = None,
                                 with_params = None):# TODO 
        with torch.no_grad():
            self.implicitnet_train(False)
            self.hypernet.eval()
            m.eval()
            if with_params is None:
                # update the mainnet based on the so-far trained hypernet then instantiate out
                if self.lowrank:
                    # forward processing of embeddings for all layers
                    self.hypernet.forward_emb(emb_in=emb_in, t= t)
                for name, param in m.named_parameters():
                    gen_params = self.hypernet.generate_W(name, emb_dict = emb_in)
                    
                    if (self.hyper_grad):
                        gen_params *= t
                        
                    if self.init_params is not None:
                        gen_params = self.init_params[name] + gen_params
                    param.data.copy_(gen_params)
                # self.forward_Hypernet("updating")
            else:
                # if params to set is given
                for name, param in m.named_parameters():
                    param.data.copy_(with_params[name])
            return m

    def marterialize_Implicitnet_grad_learning(self, m, t = None,
                                 with_params = None, lr = None):# TODO 
        updated_params = dict()
        with torch.no_grad():
            self.implicitnet_train(False)
            self.hypernet.eval()
            m.eval()
            # update the mainnet based on the so-far trained hypernet then instantiate out
            if t != 0.:
                if self.lowrank:
                    # forward processing of embeddings for all layers
                    self.hypernet.forward_emb(emb_in=None, t= t)
            for name, param in m.named_parameters():
                if t != 0.:
                    gen_params = self.hypernet.generate_W(name, emb_dict = None)
                    gen_params = with_params[name] + gen_params
                else:
                    gen_params = self.init_param_0[name]
                updated_params[name] = gen_params
                param.data.copy_(gen_params)
 
            return m, copy.deepcopy(updated_params)
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

    #################################################################################################################
    #################################################################################################################
    
    # hyper grad
    def compute_gt_grad(self, loss, lr):
        # Compute grad of loss w.r.t the generated parameters

        # iterate through layers
        
        list_of_grads = []
        list_of_params = []
        for i, (name, base, localname) in enumerate(self.name_base_localname):
            params = getattr(base, localname)
            list_of_params.append(params)
            # print("compute grad", params.shape)
        grads = torch.autograd.grad(loss, list_of_params, retain_graph=True, create_graph=False)
        list_of_grads = [g.detach().clone() * lr * torch.tensor(-1.) for i, g in enumerate(grads)] # TODO check the alignment in indexing of param in opt
        return list_of_grads, [t.detach().clone() for t in list_of_params]
    # hyper grad
    def get_generated_params(self, loss):
        # Compute grad of loss w.r.t the generated parameters

        # iterate through layers
        
        list_of_grads = []
        list_of_params = []
        for i, (name, base, localname) in enumerate(self.name_base_localname):
            params = getattr(base, localname)
            list_of_params.append(params)
        return list_of_params