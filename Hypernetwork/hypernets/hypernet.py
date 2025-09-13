import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import numpy as np
def shape_to_outdim(shape, node_direction = "W"):
    dims = len(shape)
    kernel_dim = 1
    if dims == 1:
        W = 1 if node_direction == "W" else shape[0]
        H = shape[0] if node_direction == "W" else 1
    else:
        W = shape[1] if node_direction == "W" else shape[0]    
        H = shape[0] if node_direction == "W" else shape[1]
        if dims == 4:
            # E.g., 2d Conv
            kernel_dim = shape[2] * shape[3]
        elif dims == 3:
            # E.g., 1d Conv
            kernel_dim = shape[2]
    return H, W, kernel_dim

def search_bias(dict, name):
    if name in dict:
        return True 
    else: return False
    
class Hypernet_base(nn.Module):
    def __init__(self, param_list, 
                 hidden_dim = 128, iteration = 1, 
                 node_direction = "W", lowrank = False, rank = 4, type_ = "lowrank",
                 learnable_emb = False, zero_init_emb = False, base = "mlp", 
                 cond_dim = 0, cond_emb_type = None, hyper_grad = False,
                 device = "cpu"):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(Hypernet_base, self).__init__()
        self.base = base
        self.param_list = param_list
        self.hidden_dim = hidden_dim
        self.depth = 0
        self.iteration = iteration
        self.node_direction = node_direction
        self.lowrank = lowrank
        self.rank = rank
        self.zero_init_emb = zero_init_emb
        self.layers = dict()
        self.device = device 
        self.learnable_emb = learnable_emb
        self.embedding_in = dict()
        self.emb_dict = dict()
        if not lowrank:
            self.type_ = "linear"
        else:
            self.type_ = type_
        self.set_projections()
        if not lowrank:
            self.init_embeddings()
        else:
            self.init_emb_lowrank()
        
        self.hyper_grad = hyper_grad
        self.cond_dim = cond_dim
        self.cond_emb_type = cond_emb_type
        if self.hyper_grad:
            self.init_cond_emb()
    def generate_W(self, layer_att_name, outemb = False, decom = False, emb_dict = None):
        if decom:
            raise NotImplementedError("not fully implemented yet")
        # TODO two cases where depth wise and depth included --> HOW TO BATCH PROCESSING THE HIDDEN FEATURES
        if self.param_list[layer_att_name]["local_name"] != "bias":
            self.store_bias = dict()
            layer_id = self.param_list[layer_att_name]["layer_id"] # get layer id (1~L) given attribute name
            
            if not self.lowrank:
                # retrive embeddings and forward process
                emb =self.get_emb(layer_id=layer_id, emb_dict= emb_dict)
                emb = self.forward_transformation(emb) # (1, H, dz)
            else:
                emb =self.emb_dict[f'layer_{layer_id}'] # (1, dz)
                
            # Projection
            if not decom:
                W, bias = self.projections[f"layer_{layer_id}"](emb) # (1, H, dz) OR (1, dz) --> WEIGHT & BIAS
            else:
                Uw, Vw, Sw, bias = self.projections[f"layer_{layer_id}"](emb, mode = "decom") 
            
            # Store Bias 
            if self.param_list[layer_att_name]['hasBias']:
                prefix = layer_att_name.rsplit('.weight', 1)[0]  
                self.store_bias[prefix+".bias"] = bias
                
            if outemb:
                if not self.lowrank:
                    return emb
                else:
                    return self.emb_dict
            else:
                if decom:
                    return [Uw, Vw, Sw]
                else:
                    return W # (out, in, *)
        else:
            try:
                if outemb:
                    return None
                else:  
                    if decom:
                        return self.store_bias[layer_att_name] 
                    else:
                        return self.store_bias[layer_att_name] # (out,)
            except:
                raise ValueError(f"Somehow key is mismatching or bias for '{layer_att_name}' is not available ")
    def forward_emb(self, emb_in = None, t = None):
        if emb_in is None:
            if self.learnable_emb:
                all_emb = self.all_emb 
                all_emb = all_emb if (all_emb.shape[1] == 1) or (self.base == "mlp") \
                    else all_emb + Create_SinusoidalEmb(self.depth, self.hidden_dim).unsqueeze(0).to(self.device)
                    
            else: all_emb = self.all_emb * 0.02
        else:
            if self.learnable_emb:
                all_emb = emb_in
                all_emb = all_emb if (all_emb.shape[1] == 1) or (self.base == "mlp") \
                    else all_emb + Create_SinusoidalEmb(self.depth, self.hidden_dim).unsqueeze(0).to(self.device)
            else: all_emb = emb_in * 0.02
        if (self.hyper_grad):
            assert (t is not None)
            t = t.unsqueeze(0).expand(all_emb.shape[1], -1)
            all_emb = torch.concat((all_emb, self.cond_emb(t).unsqueeze(0)), dim= -1)
            all_emb = self.fusion_layer(all_emb)
        emb = self.forward_transformation(all_emb) # (1, depth, dz) or (1,1,dz)
        if (emb.shape[1] == 1) or (self.base == "mlp"):
            # expand the depth dimension
            # emb = emb.repeat(1,self.depth,1)
            emb = emb.expand(1,self.depth,emb.shape[-1])
        self.emb_dict = dict()
        for i in range(1,self.depth+1):
            name = f"layer_{i}" 
            self.emb_dict[name] = emb[:,i-1,:] # store (1, dz)
        
    def set_projections(self):
        projections = dict()
        self.embedding_H_dim = dict()
        for name in self.param_list:
            params = self.param_list[name]
            # No need to instantiate projection layers separately 
            # for bias (which uses the one shared with weight projection)
            if params["local_name"] != "bias": 
                self.depth += 1
                self.param_list[name]["layer_id"] = self.depth
                shape = params["shape"] # e.g., (out, in, k, k) 
                H,W,K = shape_to_outdim(shape) 
                self.param_list[name]["num_emb"] = H
                
                self.layers[self.depth] = name
                # search bias 
                prefix = name.rsplit('.weight', 1)[0]
                if search_bias(self.param_list, prefix+".bias"):
                    self.param_list[name]["hasBias"] = True
                    self.param_list[prefix+".bias"]["layer_id"] = self.depth
                else: 
                    self.param_list[name]["hasBias"] = False

                # output projection
                projections[f"layer_{self.depth}"] = WeightProjectionLayer(in_dim = self.hidden_dim, 
                                                                           linear_out_dim = W * K, 
                                                                           
                                                                            H_dim = H * K, W_dim = W * K, rank = self.rank, 
                                                                            type_ = self.type_, node_direction = self.node_direction, 
                                                                            bias = self.param_list[name]["hasBias"],
                                                                            shape = shape,
                                                                            normalization=False)
                
        self.projections = nn.ModuleDict(projections)
        return self.projections

    def init_embeddings(self):
        '''Construct input "L (num. layers)" embeddings of shape (1, H_i, dz) for all i = 1, ... (outdim),
        where outdim is the output dimension of the ith layer. 
        
        
        We independently use 
        sinusoidal positional embeddings in L direction 
        +
        fourier embeddings in H direction 
        
        and concat them to get final input emb
        emb = concat(sin_emb, fourier_emb)
        
        Ituitively, 
        one may see this as  
        1) the sinusoidal emb contributes to depth-wise charactersitcs of layers
        2) the fourier embeddings contribute to (num.) output-wise characteristics

        
        ----------------
        Final input embedding shape is L number of (1, H_i, dz) for i = 1, ..., L  .
        We do depth-wise forward operation: (1, H_i, dz) -> (1, H_i, W + 1) where 1 is for bias
        '''
        h = self.hidden_dim // 4
        
        # Get embeddings along depth
        depth_emb = Create_SinusoidalEmb(self.depth, h)
        
        # Instantiate H-emb object 
        ff_emb = FourierEmb(dim = h * 3,
                   ff_sigma=2048,
                   in_dim = 1)
        
        # The input embeddings are fixed and not learnable
        for i in range(1,self.depth+1):
            H = self.param_list[self.layers[i]]["num_emb"]
          
            if self.node_direction == "H":
                if self.param_list[self.layers[i]]["hasBias"]:
                    H += 1 # bias 

            # Init embedding here
            h_emb = ff_emb.create_fouieremb(H, coord_type = "random") # (H, dz/2)
            layer_emb = depth_emb[i-1].unsqueeze(0).expand(H, -1)            
            emb = torch.concat((layer_emb,
                                h_emb), dim = -1).unsqueeze(0)
            ####################
            # emb = torch.randn((1, H, self.hidden_dim))
            name = f"layer_{i}" 
            
            self.register_buffer(name, emb)
            self.emb_dict[name] = name
        self.all_emb = None
        
    def init_emb_lowrank(self): #TODO --> CHECK THIS EMBEDDING SHAPE & HOW TO BATCH PROCESSING?
        # Instantiate H-emb object 
        ff_emb = FourierEmb(dim = self.hidden_dim,
                    ff_sigma=2048,
                    in_dim = 1)
        # print*()
        if self.base == "mlp":
            emb = ff_emb.create_fouieremb(H = 1, coord_type = "absolute").unsqueeze(0) # (1, 1, dz) shared across depth
        else:
            # Init embedding here
            emb = ff_emb.create_fouieremb(H = self.depth, coord_type = "absolute").unsqueeze(0) # (1, depth, dz)
        if not self.learnable_emb: 
            print("fixed input emb")
            if self.zero_init_emb:
                emb = torch.zeros_like(emb)
                print("zero init embeddings")
            self.register_buffer("all_emb", emb)
        else:
            print("learnable input emb")
            if self.zero_init_emb:
                emb = torch.zeros_like(emb)
                print("zero init embeddings")
            else: emb *= 0.02
            self.all_emb = nn.Parameter(emb, requires_grad= True)
    def get_emb(self, layer_id, emb_dict = None):
        return getattr(self, self.emb_dict[f'layer_{layer_id}'] if emb_dict is None else emb_dict[f'layer_{layer_id}'])

    def forward_transformation(self, emb):
        raise NotImplementedError("Specific subclass (transformation) method")
    
    def set_transformation(self, outdim, indim = None, grad_ = False):
        raise NotImplementedError("Specific subclass (transformation) method")
    
    def init_cond_emb(self):
        if self.cond_emb_type == "linear":
            self.cond_emb = nn.Linear(1, self.cond_dim)
            self.fusion_layer = nn.Linear(self.cond_dim + self.hidden_dim, self.hidden_dim)
        elif self.cond_emb_type == "fourier":
            raise NotImplementedError("")
        else:
            self.cond_emb = nn.Identity()
            self.fusion_layer = nn.Linear(1 + self.hidden_dim, self.hidden_dim)
            
def Create_SinusoidalEmb(length, dim):
    assert (dim % 2) == 0
    # Absolute positions
    position = torch.arange(length, dtype=torch.float).unsqueeze(-1) # (*, 1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (length, dim)

class FourierEmb(nn.Module):
    def __init__(self, 
                 dim,  
                 ff_sigma=64,
                 in_dim = 1, 
                 learnable = False):
        super(FourierEmb, self).__init__()      
        assert (dim % 2) == 0
        self.input_dim = in_dim # coord dimensionality
        self.dim_half = int(dim / 2)
        self.ff_sigma = ff_sigma
        
        # Gaussian ff embedding
        self.ff_linear = torch.randn(self.input_dim, self.dim_half) * self.ff_sigma
        
    def create_fouieremb(self, H, coord_type = "relative"):
        shape = [H]
        if coord_type == "relative":
            coord = shape2coordinate(shape) # (H,1)
        elif coord_type == 'random':
            coord = torch.randn((H,1), dtype=torch.float) 
        else:
            coord = torch.arange(H, dtype=torch.float).unsqueeze(-1) # (*, 1)
            
        fourier_features = torch.matmul(coord, self.ff_linear)
        fourier_features = [torch.cos(fourier_features), torch.sin(fourier_features)]
        fourier_features = torch.cat(fourier_features, dim=-1)
        return fourier_features # (H, dim)
    
def shape2coordinate(spatial_shape, min_value=-1.0, max_value=1.0, upsample_ratio=1, device=None):
    # Relative positions
    nnstates = []
    for num_s in spatial_shape:
        num_s = int(num_s * upsample_ratio)
        _nnstates = (0.5 + torch.arange(num_s, device=device)) / num_s
        _nnstates = min_value + (max_value - min_value) * _nnstates
        nnstates.append(_nnstates)
    nnstates = torch.meshgrid(*nnstates, indexing="ij")
    nnstates = torch.stack(nnstates, dim=-1)
    return nnstates # [given shape, coor_dim]

class WeightProjectionLayer(nn.Module):
    def __init__(self, in_dim, linear_out_dim, 
                 H_dim = None, W_dim = None, rank = 4, 
                 type_ = "linear", node_direction = "W", bias = False, shape = None,
                 normalization = False):
        """
        in_dim = hidden dim 
        
        out_dim = [H dim, W dim]
        """
        super(WeightProjectionLayer, self).__init__()   
        
        self.in_dim = in_dim
        self.linear_out_dim = linear_out_dim
        self.H_dim = H_dim
        self.W_dim = W_dim
        self.type_ = type_
        self.rank = rank
        self.bias_ = bias
        self.node_direction = node_direction
        self.shape = shape
        self.target_dim = len(self.shape)
        self.fullrank_proj = False
        # TODO add full rank version?
        
        if type_ == "linear": # TODO DOES IT handle single dimension -> i think yes?
            
            if bias: 
                if self.node_direction == "H":
                    raise NotImplementedError("")
                    self.projection_bias = nn.Linear(in_dim, self.shape[0], bias = False)
                elif self.node_direction == "W":
                    self.linear_out_dim = self.linear_out_dim + 1
            self.projection = nn.Sequential(nn.Linear(in_dim, self.linear_out_dim, bias = True),
                                            nn.LayerNorm(self.linear_out_dim,elementwise_affine=False) if normalization else nn.Identity())
            
        else:
            if self.target_dim == 1:
                # if target layer weight dim is 1 (e.g., layernorm), simply use linear layer as above
                if bias: 
                    self.projection_W_bias = nn.Sequential(nn.Linear(in_dim, self.shape[0], bias = True),
                                                           nn.LayerNorm(self.shape[0],elementwise_affine=False) if normalization else nn.Identity())
                self.projection_W = nn.Sequential(nn.Linear(in_dim, self.shape[0], bias = True),
                                                  nn.LayerNorm(self.shape[0],elementwise_affine=False) if normalization else nn.Identity())
            else:
                # if (self.rank > self.shape[0]) or (self.rank > self.shape[1]):
                #     # do full rank? TODO TEMPORARY FOR NOW
                #     self.rank = 2
                # assert (self.rank < self.shape[0]) and (self.rank < self.shape[1]) # TODO change it to if statement? or add conditions to if self.target_dim == 1: in case either H and/or W is already small enough
                full_params_perlayer = torch.tensor(self.shape).prod().item()
                if full_params_perlayer < 3500:
                    self.fullrank_proj = True
                    # FUll rank projection
                    self.projection_fullrank = nn.Sequential(nn.Linear(in_dim, full_params_perlayer, bias = True),
                                                             nn.LayerNorm(full_params_perlayer,elementwise_affine=False) if normalization else nn.Identity())
                    if self.bias_:
                        self.projection_bias_fullrank = nn.Sequential(nn.Linear(in_dim, self.shape[0], bias = True),
                                                                      nn.LayerNorm(self.shape[0],elementwise_affine=False) if normalization else nn.Identity())
                else:
                    if type_ == "lowrank":
                        self.projection_U = nn.Sequential(nn.Linear(in_dim, self.H_dim * self.rank, bias = True),
                                                          nn.LayerNorm(self.H_dim * self.rank,elementwise_affine=False) if normalization else nn.Identity())
                        self.projection_V = nn.Sequential(nn.Linear(in_dim, self.W_dim * self.rank, bias = True),
                                                          nn.LayerNorm(self.W_dim * self.rank,elementwise_affine=False) if normalization else nn.Identity())
                        if self.bias_:
                            self.projection_bias = nn.Sequential(nn.Linear(in_dim, self.shape[0], bias = True),
                                                                 nn.LayerNorm(self.shape[0],elementwise_affine=False) if normalization else nn.Identity())
                    
                    elif type_ == "svd":
                        self.projection_U = nn.Sequential(nn.Linear(in_dim, self.H_dim * self.rank, bias = True),
                                                          nn.LayerNorm(self.H_dim * self.rank,elementwise_affine=False) if normalization else nn.Identity())
                        self.projection_V = nn.Sequential(nn.Linear(in_dim, self.W_dim * self.rank, bias = True),
                                                          nn.LayerNorm(self.W_dim * self.rank,elementwise_affine=False) if normalization else nn.Identity())
                        self.sigma = nn.Sequential(nn.Linear(in_dim, self.rank, bias = True),
                                                   nn.LayerNorm(self.rank,elementwise_affine=False) if normalization else nn.Identity())
                        if self.bias_:
                            self.projection_bias = nn.Sequential(nn.Linear(in_dim, self.shape[0], bias = True),
                                                                 nn.LayerNorm(self.shape[0],elementwise_affine=False) if normalization else nn.Identity())
                
        # self.init_weights()
        self.apply(_weights_init)
    def forward(self, z, mode = "full"):        
        if self.type_ == "linear":
            # H or W in sequenec dimension
            # z (1, *, dz) -> out { (1, H, W), (H) }
            W = self.projection(z).squeeze() # (1, H, w (+1) ) -> (H, W (+1)) 
            # Get Bias 
            if self.bias_:
                if self.node_direction == "W":
                    bias = W[:, -1:].squeeze() # (
                    W = W[:, :-1]  # shape (H, W (* K^2))
                elif self.node_direction == "H":
                    raise NotImplementedError("")
                    bias = W[-1:, :].squeeze() # (H) # TODO tHINK ABOUT HOW TO DO BIAS FOR H MODE WITH KERNELS? --. NEED TO SAMPLE EMBEDDINGS SEPARATELY
                    W = W[:-1, :]  # shape (W, H (* K^2))
                    W = W.permute(1,0) # shape (H, W)
            else: bias = None

            W = W.view(self.shape)
            return W, bias
        
        else:
            # depth in sequence dimension 
            # z (1, 1, dz) -> out { (1, H, W), (H) }
            
            if self.target_dim != 1:
                if self.fullrank_proj:
                    W = self.projection_fullrank(z).view(self.shape)
                    if self.bias_:
                        bias = self.projection_bias_fullrank(z).view(self.shape[0])
                    else: bias = None
                    return W, bias
                else:
                    Uw = self.projection_U(z) # (1, H (* k^2) * rank )
                    Vw = self.projection_V(z) # (1, W (* k^2) * rank)
                    if self.bias_:
                        bias = self.projection_bias(z).view(self.shape[0]) # (outdim,)
                    else: bias = None
                    
                    # Uw = Uw.view(self.shape[0], self.rank, -1) # (H, rank, K^2)
                    # Vw = Vw.view(self.shape[1], self.rank, -1) # (W, rank, K^2)
                    Uw = Uw.view(-1, self.shape[0], self.rank) # (K^2, H, rank)
                    Vw = Vw.view(-1, self.shape[1], self.rank) # (K^2, W, rank)
                    # Orthogonality 
                    # Uw, _ = torch.linalg.qr(Uw, mode='reduced') # (K^2, H, rank)
                    # Vw, _ = torch.linalg.qr(Vw, mode='reduced') # (K^2, W, rank)
                     
                    if self.type_ == "svd":
                        Sw = torch.exp(self.sigma(z))
                        Sw = Sw.view(1, 1, self.rank) # (1, 1, rank)
                        SVT = Vw * Sw # (K^2, W, rank)
                    else: Sw = None

                    if mode == "full":
                        W = torch.einsum('bir,bjr->bij', Uw, SVT if self.type_ == "svd" else Vw).permute(1,2,0) # (H, W, K^2)
                        
                        return W.view(self.shape), bias
                    
                    elif mode == "decom":
                        # raise NotImplementedError() # TODO How to reshape it or maybe not needed (W,rank, *)
                        return Uw.permute(1,2,0).squeeze(), Vw.permute(1,2,0).squeeze(), \
                                Sw.permute(1,2,0).squeeze(), bias.squeeze()  
                    
            # if target layer weight dim is 1 (e.g., layernorm, learnable tokens, etc. ), simply use linear layer
            else:
                W = self.projection_W(z).view(self.shape)
                if self.bias_:
                    bias = self.projection_W_bias(z).view(self.shape[0])
                else: bias = None
                
                return W, bias
    def init_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # init.uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)