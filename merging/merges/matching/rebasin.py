from torch import nn
import torch
import copy

from tqdm import tqdm
from merges.profiles.profile import NNProfileBase
from merging.merges import utils
from merging.merges.profiles.lookup import *

class RebasinOperator:
    def __init__(self, 
                 device = None,
                 
                 sequence_model = False):
        super(RebasinOperator, self).__init__()
        self.device = device      

        # Common learnable modules that require some extra cares. 
        self.module_handler = {
                                "LayerNorm": self.LayerNorm_handler,
                                "BatchNorm2d": self.BatchNorm2d_handler,
                                "Identity": self.SkipConnection_handler, # skip connection needs to be explicitly defined as a module for merging. 
                                "Conv2d": self.Conv2d_handler,
                                
                                "Linear": self.Linear_handler,
                                "BatchNorm1d": self.Batchnorm1d_handler,
                                "Conv1D": self.Conv1D_handler,
                                "Embedding": self.Embedding_handler
        }
        self.permutation_based = False
    def rebasin(self, 
                   p: NNProfileBase,
                   id = "model A"):
        # if key in self.net_profiles[0].attention_node_id:
        # Looping through net profile 
        candidate_net = copy.deepcopy(p.net)
        self.permutation_based = p.permutation_based
        for idx, key in enumerate(tqdm(p.profile, desc = f"Rebasining {id}: ")):
            inverse_id, forward_id = p.profile[key]["transform_id"]
            inverse_att, forward_att = p.profile[key]["attention"]
            if (inverse_id is not None) or (forward_id is not None):
                # 1) Get corresponding parametric layer w
                w = utils.get_module(candidate_net, key)
                if (p.profile[key]["module_type"] in PARAMETRIC_NORMS) and (w.weight is None) and (self.permutation_based):
                    # For non-parametric nn normalization and permutation-based, pass the iter 
                    continue
                # Pre-Handle modules for re-instantiating them with equiv custom class for merging  
                if forward_id is not None:
                    w_dim = p.tranforms[forward_id]["forward"].shape[0]
                elif inverse_id is not None:
                    w_dim = p.tranforms[inverse_id]["inverse"].shape[0]
                w = self.module_handler[p.profile[key]["module_type"]](w, w_dim)

                # Some modules have weight arrangement as (input dim, output dim), yet we use the convention of (out, in)  
                if p.profile[key]["module_type"] == "Conv1D" or p.profile[key]["module_type"] == "Embedding":
                    transpose_ = True 
                else: transpose_ = False

                # W combined into T_i
                if forward_id is not None:
                    # if forward_id in p.tranforms:
                    # (TW)(X)
                    if forward_att:
                        new_weight, new_bias = self.forward_MHA_compose(w, 
                                                    outer_mf= p.outer_mh_transforms[forward_id]["forward"],
                                                    inner_mf=p.tranforms[forward_id]["forward"],
                                                    transpose=transpose_,
                                                    qk_mf = None)
                        new_running_mean, new_running_var = None, None
                    else:
                        # Conventional inner perm
                        try:
                            new_weight, new_bias, new_running_mean, new_running_var = \
                                            self.forward_compose(w, p.tranforms[forward_id]["forward"], 
                                            transpose= transpose_)
                        except:
                            print("key: ", key)
                            print("transform: ", p.profile[key]["transform_id"])
                    w = self.update(w, new_weight, new_bias, new_running_mean, new_running_var)

                # T_{i-1} combined into W
                if inverse_id is not None:
                    # if inverse_id in p.tranforms:
                    # (WT^-1)(X)
                    if inverse_att:
                        new_weight, new_bias = self.invser_MHA_compose(w, 
                                                    outer_mf= p.outer_mh_transforms[inverse_id]["inverse"],
                                                    inner_mf=p.tranforms[inverse_id]["inverse"],
                                                    transpose=transpose_)
                    else:
                        new_weight, new_bias = self.inverse_compose(w, p.tranforms[inverse_id]["inverse"], 
                                                                    transpose= transpose_)
                    w = self.update(w, new_weight, new_bias)

                # Update the actual module in the candidate net
                utils.update_module(candidate_net, key, w)
        candidate_net.train(False)
        return candidate_net
    
    def forward_compose(self, w, mf, transpose = False):
        '''
        (TW)X Forward composition
        It assumes that the weight parameters (for dim > 1) is of shape (output dim, input dim) followuing the nn.module convention
        '''
        
        # TODO for conv1d module in gpt2 (transformers lib), the weights must be transposed.
        with torch.no_grad(): 
            if w.weight is not None:
                weight_raw = w.weight.clone().detach() # (outdim ,indim)
                if transpose:
                    # weight params in Conv1D (huggingface) is originally (indim, output) 
                    weight_raw = weight_raw.T
                # Bias 
                if not hasattr(w, "bias"):
                    bias_raw_new = None
                elif w.bias is not None: 
                    bias_raw = w.bias.clone().detach()
                    bias_raw_new = torch.einsum('jk..., k... -> j...', mf, bias_raw.unsqueeze(-1)).squeeze() # 'jk..., ki... -> ji...'
                elif w.bias is None: 
                    bias_raw_new = None

                # Weights
                if weight_raw.dim() == 1: 
                    w_raw_new = torch.einsum('jk..., k... -> j...', mf, weight_raw)  # mf @ weight_raw.unsqueeze(0)

                elif weight_raw.dim() > 1:
                    try:
                        w_raw_new = torch.einsum('jk..., ki... -> ji...', mf, weight_raw) 
                    except:
                        print(w)
                        print(mf.shape)
                        print(weight_raw.shape)
                if transpose:
                    # back to (indim, output) 
                    w_raw_new = w_raw_new.T

            elif w.weight is None: 
                # This is a case of when forward compose is made on implicit identity function (i.e., None value weights) 
                w_raw_new = mf.detach().clone()
                if transpose:
                    # (indim, output) 
                    w_raw_new = w_raw_new.T

                bias_raw_new = None

            if hasattr(w, "running_mean") and self.permutation_based:
                running_mean_new = torch.einsum('jk..., k... -> j...', mf, w.running_mean)
                running_var_new = torch.einsum('jk..., k... -> j...', mf, w.running_var)
            else:
                running_mean_new, running_var_new = None, None
        return w_raw_new, bias_raw_new, running_mean_new, running_var_new

    def inverse_compose(self, w, mf, transpose = False):
        '''
        (WT^-1)X Inverse composition
        It assumes that the weight parameters (for dim > 1) is of shape (output dim, input dim) followuing the nn.module convention
        '''
        with torch.no_grad(): 
            if w.weight is not None:
                weight_raw = w.weight.clone().detach() 
                if transpose:
                    # weight params in Conv1D (huggingface) is originally (indim, output) 
                    weight_raw = weight_raw.T

                if not hasattr(w, "bias"):
                    bias_raw_new = None
                elif w.bias is not None:
                    # Nothing to do with bias terms for inverse.
                    bias_raw = w.bias.clone().detach()
                    bias_raw_new = bias_raw
                elif w.bias is None: 
                    bias_raw_new = None

                if weight_raw.dim() == 1:
                    w_raw_new = torch.einsum('jk..., k... -> j...', weight_raw, mf) 
                elif weight_raw.dim() > 1:
                    w_raw_new = torch.einsum('jk..., ki... -> ji...', weight_raw, mf) 

                if transpose:
                    # back to (indim, output) 
                    w_raw_new = w_raw_new.T

            elif w.weight is None:
                w_raw_new = mf.detach().clone()
                if transpose:
                    # (indim, output) 
                    w_raw_new = w_raw_new.T
                bias_raw_new = None
        return w_raw_new, bias_raw_new
    
    def forward_MHA_compose(self, c_attn,
                            outer_mf, inner_mf, 
                            transpose, qk_mf = None):
        w_qkv = c_attn.weight.clone().detach()
        w_qkv = list(w_qkv.split(w_qkv.shape[1] // 3, dim = 1)) # shape (indim, outdim) for huggingface conv1D
        if c_attn.bias is None:
            b_qkv = [None, None, None]
        else:
            b_qkv = c_attn.bias.clone().detach()
            b_qkv = list(b_qkv.split(b_qkv.shape[0] // 3, dim = 0))
        
        # outer perm forward composition on q, k, and v  (merge)
        for i, (w,b) in enumerate(zip(w_qkv,b_qkv)):
            w_qkv[i], b_qkv[i] = self.forward_outer(w, outer_mf, b, transpose= transpose)

        # inner perm forward composition on v
        w_qkv[-1], b_qkv[-1] = self.forward_inner(w_qkv[-1], inner_mf, b_qkv[-1], transpose= transpose)

        # (optional) inner perm forward composition on q and k
        if qk_mf is not None:
            raise NotImplementedError("not supported yet")
            for i, (w,b) in enumerate(zip(w_qkv[:-1],b_qkv[:-1])):
                w_qkv[i], b_qkv[i] = self.forward_inner(w, qk_mf, b, transpose= transpose)
        
        w_qkv = torch.cat((w_qkv), dim = 1)
        b_qkv = torch.cat((b_qkv), dim = 0) if b_qkv[0] is not None else None

        return w_qkv, b_qkv
    def invser_MHA_compose(self, attn_proj,
                            outer_mf, inner_mf, 
                            transpose):
        w = attn_proj.weight.clone().detach()
        if attn_proj.bias is None:
            b = None
        else:
            b = attn_proj.bias.clone().detach()
        # outer
        w, b = self.inverse_outer(w, outer_mf, b, transpose= transpose)
        # inner 
        w, b = self.inverse_inner(w, inner_mf, b, transpose= transpose)
        return w, b

    def forward_inner(self, w, mf, b = None, transpose = False):
        '''Within-head permutation inverse composition for each head (inner)'''
        with torch.no_grad(): 
            num_head = mf.shape[0]
            
            # Bias 
            if b is not None: 
                bias_raw = b
                bias_dim = bias_raw.shape[0]
                bias_raw = bias_raw.reshape(num_head, bias_dim // num_head)
                bias_raw_new = torch.einsum('hjk..., hk... -> hj...', mf[:,:,:], bias_raw[:,:]).view(bias_dim)
            else: bias_raw_new = None

            # Weights
            weight_raw = w # (outdim ,indim)
            if transpose:
                # weight params in Conv1D (huggingface) is originally (indim, output) 
                weight_raw = weight_raw.T

            out_dim = weight_raw.shape[0]

            if weight_raw.dim() == 1: 
                weight_raw = weight_raw.reshape(num_head, out_dim // num_head)
                w_raw_new = torch.einsum('hjk..., hk... -> hj...', mf[:,:,:], weight_raw[:,:]).view(out_dim)
           
            elif weight_raw.dim() > 1:
                weight_raw = weight_raw.reshape(num_head, out_dim // num_head, -1)
                w_raw_new = torch.einsum('hjk..., hkd... -> hjd...', mf[:,:,:], weight_raw[:,:,:]).view(out_dim, -1)

            if transpose:
                # back to (indim, output) 
                w_raw_new = w_raw_new.T
        return w_raw_new, bias_raw_new
    
    def inverse_inner(self, w, mf, b = None, transpose = False):
        '''Within-head permutation inverse composition for each head (inner)'''
        with torch.no_grad(): 
            num_head = mf.shape[0]
            
            # Bias 
            if b is not None: 
                bias_raw = b
                bias_raw_new = bias_raw
            else: bias_raw_new = None

            # Weights
            weight_raw = w # (outdim ,indim)
            if transpose:
                # weight params in Conv1D (huggingface) is originally (indim, output) 
                weight_raw = weight_raw.T

            in_dim = weight_raw.shape[1]

            if weight_raw.dim() == 1: 
                weight_raw = weight_raw.reshape(num_head, in_dim // num_head)
                w_raw_new = torch.einsum('hk..., hkj... -> hj...', weight_raw[:,:], mf[:,:,:]).reshape(in_dim)
           
            elif weight_raw.dim() > 1:
                weight_raw = weight_raw.reshape(-1, num_head, in_dim // num_head)
                w_raw_new =  torch.einsum('dhk..., hkj... -> dhj...', weight_raw[:,:,:], mf[:,:,:]).reshape(-1, in_dim)

            if transpose:
                # back to (indim, output) 
                w_raw_new = w_raw_new.T
        return w_raw_new, bias_raw_new

    def forward_outer(self, w, head_t, b = None, transpose = False):
        '''Head permutation forward composition (outer)''' 
        num_head = head_t.shape[0]
        with torch.no_grad(): 
            # Bias 
            if b is not None: 
                bias_raw = b
                bias_dim = bias_raw.shape[0]
                bias_raw = bias_raw.reshape(num_head, bias_dim // num_head)
                bias_raw_new = torch.einsum('jk..., k... -> j...', head_t, bias_raw).reshape(bias_dim)
            else: bias_raw_new = None

            # Weights
            weight_raw = w # (outdim ,indim)
            if transpose:
                # weight params in Conv1D (huggingface) is originally (indim, output) 
                weight_raw = weight_raw.T
            out_dim = weight_raw.shape[0]

            weight_raw = weight_raw.reshape(num_head, out_dim // num_head, -1)
            w_raw_new = torch.einsum('jk..., k... -> j...', head_t, weight_raw).reshape(out_dim, -1)
            if transpose:
                # back to (indim, output) 
                w_raw_new = w_raw_new.T
            return w_raw_new, bias_raw_new

    def inverse_outer(self, w, head_t, b = None, transpose = False):
        '''Head permutation inverse composition (outer)'''
        num_head = head_t.shape[0]
        with torch.no_grad(): 
            # Bias 
            if b is not None: 
                bias_raw = b
                bias_raw_new = bias_raw
            else: bias_raw_new = None

            weight_raw = w # (outdim ,indim)
            if transpose:
                # weight params in Conv1D (huggingface) is originally (indim, output) 
                weight_raw = weight_raw.T # (outdim ,indim)
            in_dim = weight_raw.shape[1]
            weight_raw = weight_raw.reshape(-1, num_head, in_dim // num_head)
            w_raw_new = torch.einsum('dhk..., hj... -> djk...', weight_raw, head_t).reshape(-1, in_dim)
            if transpose:
                # back to (indim, output) 
                w_raw_new = w_raw_new.T
        return w_raw_new, bias_raw_new

    def update(self, 
               w, 
               new_weight, 
               new_bias,
               new_mean = None,
               new_var = None):
        # TODO: add concating back for kqv weights  
        with torch.no_grad(): 
            if (w.weight is None) and (new_weight is not None):
                # These merged modules do not have bias term (bias remains "None")
                del w.weight
                weight = nn.Parameter(new_weight)
                w.register_parameter("weight", weight) 
            else:
                w.weight.data.copy_(new_weight.clone().detach())
                if not hasattr(w, "bias"):
                    pass
                elif (w.bias is not None) and (new_bias is not None):
                    w.bias.data.copy_(new_bias.clone().detach())

            if (new_mean is not None) and (new_var is not None):
                w.running_mean.copy_(new_mean.clone().detach())
                w.running_var.copy_(new_var.clone().detach())
        return w
    
    def LayerNorm_handler(self, w, w_dim = None):
        '''
        This instantiates new (custom) layernorm where the scale param is in n by n matrix form.
        This is only done when "non"-permutation-based matching transforms used.  
        '''
        # this is only done when it is not permutation based 
        if not self.permutation_based:
            w_custom = utils.LayerNorm(w.weight.shape if w.weight is not None else None, 
                                       initial_affine = True if w.weight is not None else False
                                       ).to(self.device)
            # Init the custom with the original params
            with torch.no_grad(): 
                if w.weight is not None: w_custom.weight.data.copy_(torch.diag(w.weight.data.clone().detach()))
                if w.bias is not None: w_custom.bias.data.copy_(w.bias.data.clone().detach())
            return w_custom
        # TODO update graph node as well!
        else: return w

    def BatchNorm2d_handler(self, w, w_dim = None):
        '''
        This instantiates new (custom) batchnorm2d where the scale param is in matrix form.
        This is only done when "non"-permutation-based matching transforms used.
        '''
        if not self.permutation_based:
            w_custom = utils.BatchNorm2d(w.running_mean.shape, 
                                       initial_affine = True if w.weight is not None else False
                                       ).to(self.device)
            # Init the custom with the original params
            with torch.no_grad(): 
                if w.weight is not None: w_custom.weight.data.copy_(torch.diag(w.weight.data.clone().detach()))
                if w.bias is not None: w_custom.bias.data.copy_(w.bias.data.clone().detach())            
                w_custom.running_mean.copy_(w.running_mean.data.clone().detach())
                w_custom.running_var.copy_(w.running_var.data.clone().detach())
            return w_custom
            # TODO update graph node as well!
        else: return w

    def SkipConnection_handler(self, w, w_dim = None):
        '''
        Re-instantiating skip connection with a custom one that 
        has initially-None weight and bias registered.

        The identity skip connection becomes nn.Linear object with weight params only 
        '''
        w_custom = utils.SkipConnection_P(w_dim, w_dim).to(self.device)
        return w_custom
        
    def Conv2d_handler(self, w, w_dim = None):
        return w
    def Conv1D_handler(self, w, w_dim = None):
        '''Conv1D module (huggingface transformers) is different from Conv1d (nn.module)'''
        return w
    def Linear_handler(self, w, w_dim = None):
        return w
    def Batchnorm1d_handler(self, w, w_dim = None):
        raise NotImplementedError("Implement one like the custom batchnorm2d in utils")
    def Embedding_handler(self, w, w_dim = None):
        return w