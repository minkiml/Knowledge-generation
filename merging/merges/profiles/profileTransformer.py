
'''
Profile class for Transformers and LoRAs from huggingface. 
This profile is not supported for non-huggingface transformers.

Extracting the profile from transformers is done differently from the general.
It is done by using architectural features. 

1. Skip connection is implicit

2. Forward operation is done by 
    1. getting parametric attributes (weight & bias)
    2. torch.addmm to do innerproduct --> i.e., this is the true node
'''
import numpy as np
import torch
import torch.nn as nn
from transformers.utils import fx as trf_FX
from merging.merges.profiles.lookup import *
from merging.merges.profiles.profile import NNProfileBase
from merging.merges import utils

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class SyncGraphModule(trf_FX.GraphModule):
    # TODO how to override it with just different parent class ?
    '''
    This synchronizes the modified and recompiled the candidate network wrapped in graph trace with the original network 
    for class-specific methods and attributes that are non-nn.Modules. 
    '''
    def __init__(self, original_class, graph, root):
        super(SyncGraphModule, self).__init__(root, graph)
        # Copy all attributes from the original class
        for attr_name, attr_value in original_class.__dict__.items():
            if not attr_name.startswith('_') and not hasattr(self, attr_name):
                # Skip nn.Module attributes as they're already part of the root
                if isinstance(attr_value, nn.Module):
                    continue
                setattr(self, attr_name, attr_value)

        # Copy all custom methods from the original class
        for method_name in dir(original_class):
            if callable(getattr(original_class, method_name)) and not method_name.startswith('_'):
                method = getattr(original_class, method_name)
                if hasattr(method, "__get__"):  # Only copy bound methods
                    setattr(self, method_name, method.__get__(self))

class NNProfileLM(NNProfileBase):
    '''
    Profile class for GPT2.
    This class should work for other huggingface transformer models with model-specific vars known. 

    Note that some of important variables' (nodes) name should be known in prior. 
    '''
    fx = trf_FX
    GraphModule = trf_FX.GraphModule
    Node = trf_FX.Node
    def __init__(self, 
                 model,
                 network,
                 device = 'cpu',
                 permutation_based = True,
                 embeddings = False,
                 prec_residuals = False
                 ):
        super(NNProfileLM, self).__init__(model, network, device, 
                                          permutation_based, embeddings, prec_residuals)
        
        # Huggingface gpt2-specific vars 
        VARS = {"gpt2": GPT2_VARS,
                "llama": LLAMA_VARS
                }[network]
        
        self.residual_keys = VARS["residual_keys"] # ["resid_dropout", "dropout"] # keywords to recognize residual nodes.
        self.attention_key_entry = VARS["attention_key_entry"] #"attn.c_attn" # keywords to recognize entry attention nodes.
        self.attention_key_exit = VARS["attention_key_exit"] #"attn.c_proj" # keywords to recognize exit attention nodes.
        self.stage = VARS["stage"] # this is the level sub class modules where skip connection is explicitized
        
        self.create_profile()

    def ignore_attention_perms(self):
        '''Remove all the mering (MHA) transforms  assigned to attention layers '''
        for key in self.profile:
            u, v = self.profile[key]["transform_id"]
            if u in self.attention_node_id:
                u = None
            if v in self.attention_node_id:
                v = None
            self.profile[key]["transform_id"] = (u,v)

    def add_hooks(self): # TODO: why does it output booleans?? instead of values while spitting out valued outputs?
        """ 
        Add new hooks. 
        
        Note that Transformers from huggingface use its own linear module named "Conv1D" 
        while the actual forward operation is done separately with addmm operation.  
        """
        # Intermediate feature extraction 
        # best to be called when computing intermediate features
        self.clear_hooks()
        def pre_hook(m, input, id):
            in_ = input[0].detach().to(self.device) # TODO: consider managing feature shape here
            self.int_features[id] = in_
            return None
        for key in self.transform_key: # TODO
            node = self.merge_key_map[key]
            m = self.net.get_submodule(node)
            self.hooks.append(m.register_forward_pre_hook(lambda m, 
                                                          input, 
                                                          id=key: 
                                                          pre_hook(m, input, id)))
            
class NNProfileLoRA(NNProfileBase):
    '''
    Profile class with lora-specific functions.
    '''
     # TODO implement ** 
    def __init__(self, 
                 network: nn.Module,
                 device = 'cpu',
                 permutation_based = True
                 ):
        super(NNProfileLoRA, self).__init__()
        raise NotImplementedError("")
    

if __name__ == "__main__":
    from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, LlamaConfig, LlamaModel
    import random
    # seed = 42
    config = GPT2Config.from_pretrained("gpt2")
    config.n_layer = 1
    model = GPT2Model(config)
    # print(config)
    # config = LlamaConfig(num_hidden_layers = 1)
    # print(config)
    # # Initializing a model from the llama-7b style configuration
    # model = LlamaModel(config)

    def quick_model_inf(m, model_name = "Network"):
        print(f"Model: {model_name}")
        total_param = 0
        for name, param in m.named_parameters():
            num_params = param.numel()
            total_param += num_params
            print(f"{name}: {num_params} parameters")
        
        print(f"Total parameters in {model_name}: {total_param}")
        print("")
        return total_param
    quick_model_inf(model, "gpt")
    print(model)
    p = NNProfileLM(model,
                    network=config.model_type,
                    embeddings= False,
                    prec_residuals = False)

    p.add_hooks()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    sentence = "The model's outputs"
    x = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        y = p.net(**x)
    print(y["last_hidden_state"])
    xx = p.get_intermediate_features()
    i = 0
    for key in xx:
        i += 1
        print(key)
        print(xx[key].shape)
    print(i)
    p.clear_features()
    p.clear_hooks()