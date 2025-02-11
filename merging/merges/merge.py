'''
The current merge works for two candidates and not support more than 2 yet. 
'''


import torch
import copy
import os
import yaml
import torch.nn as nn
import transformers

from tqdm import tqdm
from typing import Union

from merges.profiles import profile
from merges.profiles.profileTransformer import NNProfileLM

from merges.matching import diff_matching_functions, metrics, matching_functions
from merges.matching.rebasin import RebasinOperator
from merges import utils

from transformers import AutoModel
from torch.utils.data import DataLoader


def merge(
    model_A: Union[nn.Module, str], model_B: Union[nn.Module, str],
    data_A: Union[DataLoader, str], data_B: Union[DataLoader, str],
    device = "cpu",
    config_dir = None
    ) -> nn.Module:
    # TODO: Create yml for all exp (and datasets) instead of passing the datasets. 

    # current_dir = os.path.dirname(__file__)
    # yaml_path = os.path.join(current_dir, "config.yml")
    with open("./merges/configs/gpt2merge_config.yml" if config_dir is None else config_dir) as file:
        defaults = yaml.safe_load(file)

    config = {**defaults, **{k: v for k, v in locals().items() if v is not None}}

    # Handle string paths for HuggingFace models
    if isinstance(model_A, str):
        model_A = AutoModel.from_pretrained(model_A)
    if isinstance(model_B, str):
        model_B = AutoModel.from_pretrained(model_B)
    
    # Check if models are instances of nn.Module
    if isinstance(model_A, nn.Module) and isinstance(model_B, nn.Module):
        # Compare model architectures by string representation
        if str(model_A) != str(model_B):
            raise ValueError("Models have different architectures")

        # Compare number of parameters
        params_A = sum(p.numel() for p in model_A.parameters())
        params_B = sum(p.numel() for p in model_B.parameters())
        if params_A != params_B:
            raise ValueError("Models have different number of parameters")

    if isinstance(data_A, str):
        raise NotImplementedError("")
        data_A = 1 # Load data using config.data_path and data_A
    if isinstance(data_B, str):
        raise NotImplementedError("")
        data_B = 1 # Load data using config.data_path and data_B

    merge_data = utils.merge_dataloader(data_A, data_B, # "Training" dataset A and B
                                        batch_size = config.batch, 
                                        subset_p = config.subset_portion)

    merger = ModelMerger(model_A, 
                         model_B,
                        network = config.network,
                        dataset = merge_data,
                        device = device,
                        batch_for_sim = config.batch_for_sim,
                        
                        align = config.align,
                        align_method = config.align_method,
                        align_type = config.align_type,
                        permutation_based = config.permutation_based,
                        embeddings= config.embeddings,
                        prec_residuals = config.prec_residuals,
                        qk_perm= config.qk_perm,
                        
                        merge_method = config.merge_method
                        
                        )
    
    merged_model, rebasined_model_A, rebasined_model_B = merger.merge()

    # TODO: do we make validation and analysis here?

    return merged_model, rebasined_model_A, rebasined_model_B

class ModelMerger:
    def __init__(self, 
                 # General args
                *candidate_nets, 
                network = "Lenet",
                dataset = None,
                device = None,
                batch_for_sim = 32,

                # Alighment args
                align = False,
                align_method  = "Permutation",
                align_type  = "AM", # or WM
                permutation_based = True,
                embeddings = False,
                prec_residuals = False,
                qk_perm = False, # inner MHA permutation
                
                
                # Merging args
                merge_method = "average"
                 
                 ):
        super(ModelMerger, self).__init__()
        self.qk_perm = qk_perm
        self.device = device
        self.embeddings = embeddings
        self.prec_residuals = prec_residuals
        self.network = network
        self.batch_for_sim = batch_for_sim
        self.candidate_nets = candidate_nets # Note that currently, netprofile creates a copy of the original net
        self.num_candidates = len(self.candidate_nets)
        assert self.num_candidates < 3, "No more than 2-models merging is supported yet"

        # Align args
        self.dataset = dataset
        self.alignment = align
        self.align_method = align_method
        self.align_type = align_type
        self.permutation_based = permutation_based

        # Merge args
        self.merge_method = merge_method
        
    def merge(self):

        if self.alignment:
            # These are the rebasined copy of the original nets 
            model_A, model_B = self.align(self.dataset)

        else:
            model_A, model_B = self.candidate_nets

        self.merge_method = {
            "average": Interpolation
                             }[self.merge_method]

        return self.merge_method(model_A, model_B, alpha = 0.5), model_A, model_B
    
    def align(self, dataset = None):
        '''
        Given model A and model B, this puts A and B in a common space, 
        by finding a functional-preserving transform for every parametric layers.
        
        1. Activation matching - (AM) 
        2. Weight matching - (WM)

        The resulting rebasined models are expected to be linearly connected over the domain U(A,B).
        Thus, the linear merging (e.g., averaging) will work.  
        '''
        transformers = ["gpt2", "llama", "huggingface_trf"]
        if self.network in transformers:
            PROFILER = NNProfileLM
            self.num_heads = self.candidate_nets[0].config.n_head
            if self.network == "huggingface_trf":
                self.network = self.candidate_nets[0].config.model_type
        elif self.network == "resnet":
            PROFILER = profile.NNProfileResnet
        else:
            PROFILER = profile.NNProfileGeneral
        self.net_profiles = [PROFILER(net,
                                      self.network,
                                    self.device,
                                    permutation_based = self.permutation_based,
                                    embeddings=self.embeddings,
                                    prec_residuals = self.prec_residuals
                                    ) for net in tqdm(self.candidate_nets, desc= "Extracting network profile for alignment: ")]
        
        self.net_profiles[0].print_()
        self.num_candidates = len(self.net_profiles)

        if self.align_method == "CCA":
            self.permutation_based = False
        # elif self.align_method in ["Permutation", "zipit"]:
        #     self.permutation_based = True

        for p in self.net_profiles:
            p.permutation_based = self.permutation_based

        # AM
        if self.align_type == "AM":
            try:
                self.matching_function = {
                    "Permutation": matching_functions.match_tensors_permute, # Ref
                    "CCA": matching_functions.match_tensors_zipit, # Ref
                    "zipit": matching_functions.match_tensors_cca # Ref
                    }[self.align_method]
                if self.network in transformers:
                    self.attention_matching = matching_functions.match_tensors_permute_MHA
                else: self.attention_matching = None
            except:
                raise KeyError("The argumented align mentod is not available in Activation Matching ")
            
            assert dataset is not None, "dataset (dataloader) must be passed in for AM"

            self.activation_matching(dataset)

            RO = RebasinOperator(self.device)

            # The rebasined models are new instances and don't share the same memory as the original candidate models.
            rebasind_models = [RO.rebasin(p, id = i) for i, p in enumerate(self.net_profiles)]
        
        # WM
        elif self.align_type == "WM":
            raise NotImplementedError("")
            try:
                self.matching_function = {
                    "Permutation": diff_matching_functions.Rebasinnet(), # Ref
                    "SVD": 2 # Ref
                    }[self.align_method]
            except:
                raise KeyError("The argumented align mentod is not available in Weight Matching")
        
        return rebasind_models
    
    def activation_matching(self, dataset):
        '''Compute intermediate features, similarity, and matching transform ''' 
        self.matching_metric = dict()

        for p in self.net_profiles:
            p.net.train(False)
            p.add_hooks()
            p.clear_features()
        numel = 0 # counts total num samples
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataset, 
                                                desc = f"Computing similarity features: ")):
                features = []
                if isinstance(batch, transformers.tokenization_utils_base.BatchEncoding):
                    x = batch["input_ids"]
                    attn_mask = batch["attention_mask"] if 'attention_mask' in batch else None
                    for p in self.net_profiles:
                        p.net(input_ids = x, attention_mask = attn_mask)
                        features.append(p.get_intermediate_features())
                        p.clear_features()
                else:
                    x,_ = batch
                    x = x.to(self.device)
                    for p in self.net_profiles:
                        p.net(x)
                        features.append(p.get_intermediate_features())
                        p.clear_features()
                numel += x.shape[0]
                # Computing similarity is memory intensive for large mini-batch size (with large embedding size and large spartial or sequence dim)
                # sub-split the mini-batch
                batch_size = x.shape[0]
                N = 1 if batch_size < self.batch_for_sim else batch_size // self.batch_for_sim 
                assert (batch_size % N) == 0, "Set the batch size of the dataloader used in merging to be divisible by 32"

                for j, key in enumerate(features[0]):
                    if key in self.matching_metric: pass
                    else:  
                        self.matching_metric[key] = metrics.Pairwise_covariance(correlation= False)
                    
                    chunks_all = [torch.chunk(F[key], N, dim=0) for F in features]

                    for chunk_features in zip(*chunks_all):
                        self.matching_metric[key].update(list(chunk_features), 
                                                         class_ = p.profile[p.merge_key_map[key]]["module_type"]) 
        
        for p in self.net_profiles:
            p.clear_hooks()
            p.clear_features()

        for i, key in enumerate(tqdm(self.matching_metric, desc = "Computing matching (rebasin) transforms: ")):
            C = self.matching_metric[key].finalize(numel).detach()
            if self.align_method == "cca": forw, inv = self.matching_function(C) 
            # TODO: delete
                # forw, inv, forw_A, inv_A = self.matching_f(C, dual = True) #
            else: 
                if key in self.net_profiles[0].attention_node_id:
                    # MHA permutation
                    # Outer permutation
                    assert self.attention_matching is not None
                    # outer and inner MHA matching takes in input intermediate features to W_projection
                    forw, inv, attn_head_perm = self.attention_matching(n_heads = self.num_heads, 
                                                                        permute_heads=True,
                                                                        correlation_matrix=C)
                    
                    # Inner qk matching takes in the output intermediate features from Q and K
                    if self.qk_perm:
                        raise NotImplementedError("Need to implement")
                        # TODO: Compute Q K permutation here
                        # TODO: the correlation C must be from "output" intermediate features of xQ and xK (or simply from either xK or xQ)
                        forw, inv, attn_head_perm = self.attention_matching(n_heads = self.num_heads, 
                                                                        permute_heads=True, head_assignments = attn_head_perm,
                                                                        correlation_matrix=C
                                                                        )
                       
                else:
                    forw, inv = self.matching_function(C)
                forward_ = forw.chunk(self.num_candidates, dim=1)
                inverse_ = inv.chunk(self.num_candidates, dim=0)

            '''
            The naming ("inverse" and "forward") is with respect to how the merging transform is ... ,
            rather than how it is applied. E.g., forward combine (prop_forward function in zipit codes) is done with our naming "inverse" transform.
            '''
            for i, p in enumerate (self.net_profiles):
                if key in self.net_profiles[0].attention_node_id:
                    forward, inverse, head_perm, head_perm_inv = matching_functions.get_head_perm(forward_[i], inverse_[i], 
                                                                                                          torch.arange(len(attn_head_perm)).to(self.device) 
                                                                                                          if i == 0 else attn_head_perm)
                    p.outer_mh_transforms[key] = {"inverse": head_perm_inv,
                                                  "forward": head_perm}
                    # TODO: store inner permutations for Q K here 
                    # ...
                    # ...
                else:
                    forward, inverse = forward_[i], inverse_[i]
                p.tranforms[key] = {
                                    "inverse": inverse,
                                    "forward": forward
                                    } 
    
    # hooks and intermediate features
    def clear_hooks(self):
        for p in self.net_profiles:
             p.clear_hooks()
    def clear_features(self):
        for p in self.net_profiles:
             p.clear_features()
    
    def get_data(self):
        return self.net_profiles[0].tranforms, self.net_profiles[1].tranforms, self.matching_metric



#################################
# Merge methods

def Interpolation(net_A, net_B,
                alpha = 0.5):
    # TODO implement partial merging
    state_dict_A = net_A.state_dict()
    state_dict_B = net_B.state_dict()
    state_dict_C = copy.deepcopy(state_dict_A)
    for key in state_dict_A:
        state_dict_C[key] = ((1 - alpha) * state_dict_A[key]) + (alpha * state_dict_B[key])
    
    
    intp_model = copy.deepcopy(net_A)
    intp_model.load_state_dict(state_dict_C)
    return intp_model

def Task_arithmetic():
    raise NotImplementedError

def TIES():
    raise NotImplementedError




if __name__ == "__main__":
    from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
    import torch
    from torch.utils.data import DataLoader, Dataset
    import transformers
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
    # Custom Dataset for tokenized text samples
    class TextDataset(Dataset):
        def __init__(self, sentences, tokenizer):
            self.sentences = sentences
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            # Tokenize the sentence at the given index
            return self.tokenizer(self.sentences[idx], return_tensors="pt", padding=True, truncation=True)

    config = GPT2Config.from_pretrained("gpt2")
    config.n_layer = 1
    model_A = GPT2Model(config)

    config2 = GPT2Config.from_pretrained("gpt2")
    config2.n_layer = 1
    model_B = GPT2Model(config2)

    state_dict_A = model_A.state_dict()
    state_dict_B = model_B.state_dict()
    # state_dict_C = copy.deepcopy(state_dict_A)
    for key in state_dict_A:
        print(key)
        if torch.allclose(state_dict_A[key][0], state_dict_B[key][0], rtol = 1e-4):
            print("PASS")
        else:
            print("Fail")
        print("################################# \n")

    # Step 2: Create a batch of sentences
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    sentences = [f"Sample sentence {i}" for i in range(16)]  # Example with 16 samples
    # xx = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    # print(xx)

    # Step 3: Create a Dataset and DataLoader
    dataset = TextDataset(sentences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)  # Use batch_size=4 for demonstration


    batch = 16
    subset_portion = 0.4
    network = config.model_type
    device = "cpu"
    batch_for_sim = 32
    align = True
    align_method = "Permutation"
    align_type = "AM"
    permutation_based = True
    embeddings = True
    prec_residuals = False
    qk_perm = False
    merge_method = "average"

    # merge_data = utils.merge_dataloader(model_A, model_B, # "Training" dataset A and B
    #                                 batch_size = batch, 
    #                                 subset_p = subset_portion)

    merger = ModelMerger(model_A, 
                         model_B,
                        network = network,
                        dataset = dataloader,
                        device = device,
                        batch_for_sim = batch_for_sim,
                        
                        align = align,
                        align_method = align_method,
                        align_type = align_type,
                        permutation_based = permutation_based,
                        embeddings= embeddings,
                        prec_residuals = prec_residuals,
                        qk_perm= qk_perm,
                        
                        merge_method = merge_method
                        
                        )
    
    merged_model, rebasined_model_A, rebasined_model_B = merger.merge()
    print(merged_model)
    quick_model_inf(merged_model)

    state_dict_A = model_B.state_dict()
    state_dict_B = rebasined_model_B.state_dict()
    # state_dict_C = copy.deepcopy(state_dict_A)
    for key in state_dict_A:
        print(key)
        if torch.allclose(state_dict_A[key][0], state_dict_B[key][0], rtol = 1e-4):
            print("EQUAL")
        else:
            print("DIFF")
        print("################################# \n")
