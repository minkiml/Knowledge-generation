

import torch.nn as nn
import transformers
'''
While nn.Identity itself is not parametric, 
layers with nn.Identity (e.g., skip connection) need to be parametric after merging   

Importantly, all modyles here will be considered for merging regardless whether they initially have learnable params or not. 

Note:
1. nn.Embedding is not accounted for merging. 

'''
PARAMETRIC_MODULE = (nn.Linear, nn.Conv2d, nn.Identity,
                    nn.LayerNorm, nn.BatchNorm2d, 
                    transformers.Conv1D,
                    transformers.models.llama.modeling_llama.LlamaRMSNorm # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
                    )
# Centroids for merging around each node 
PARAMETRIC_MERGING_MODULE = (nn.Linear, nn.Conv2d, transformers.Conv1D)
PARAMETRIC_MODULE_ATT = (nn.Parameter, nn.ParameterList, nn.ParameterDict) 



'''
Special modules are those that "participate" in merging but DO NOT COMPUTE its own merging transform  
(e.g., identity layer in skip connection)

Modules defined here all need to be defined in " PARAMETRIC_MODULE "
'''
SPECIAL_MODULE = (nn.Identity)



'''
All the parametric normalization modules. 
The intermediate matching tranforms to these modules are not necessary to be computed for merging,
since the normalization operation is data-dependent and not compositional. 
'''
PARAMETRIC_NORMS = ["LayerNorm",
                    "BatchNorm2d",
                    "LlamaRMSNorm"]





'''
Network-specific vars (add you own ones if necessary)

If you can define the correct values for below vars, profile of any network can be easily extracted.
'''

DEFAULT_VARS = {"stage": 1,
                "residual_keys": "default",  # No longer need this

                "attention_key_entry": "default",
                "attention_key_exit": "default"
                }

RESNET_VARS = {"stage": 1,
                "residual_keys": "downsample",

                "attention_key_entry": "default",
                "attention_key_exit": "default"
                }

GPT2_VARS = {"stage": 2,
                "residual_keys": "residual", # ["resid_dropout", "dropout"],

                "attention_key_entry": "attn.c_attn",
                "attention_key_exit": "attn.c_proj"
                }


LLAMA_VARS = {"stage": 2,
                "residual_keys": "residual", #["o_proj", "down_proj"],  # To find this, one needs to inspect through the trace.graph and finds "add" nodes representing implicit residuals 

                "attention_key_entry": "self_attn",
                "attention_key_exit": "o_proj"
                }