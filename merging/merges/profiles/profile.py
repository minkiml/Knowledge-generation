"""
Either 
wrap each model into a class and define all the necessary info for merging in there 
or 
extract the infomation about the model only (as profile) and access the models for merging through it. 


A challenge in this is how to deal with (detacting) the special but abstract layers like skip connection.

    If a model involve implicit parametric operation (e.g., skip connection is inner product with indentity matrix)
    and the operationas are involved in merging process, then, ... 

    0). Find the implicit operation in the instantiated NN object or modify the original function
    1). Explicitly define the implicit operation as a class object in the NN object. 

This is base profile class
TODO: implement specialized child class of NNProfileBase that handles specific and more complex networks.
"""
import torch
import torch.nn as nn
import numpy as np
from torch import fx as torch_FX
from merging.merges.profiles.lookup import *
from merging.merges import utils

class NNProfileBase:
    '''
    Assumes that the models are already with explicitly defined 
    originally-implicit layers (e.g., skip connection), 
    otherwise, they are not accounted for.
    '''
    fx = torch_FX
    GraphModule = torch_FX.GraphModule
    Node = torch_FX.Node
     # TODO implement selective merging ** 
    def __init__(self, 
                 model: nn.Module,
                 network = "",
                 device = 'cpu',
                 permutation_based = True,

                 embeddings = False,
                 prec_residuals = False
                 ):
        super(NNProfileBase, self).__init__()

        self.net = model
        self.network = network
        self.device = device
        self.permutation_based = permutation_based
        self.embeddings = embeddings
        self.prec_residuals = prec_residuals
        ''' Implementation-specific vars (default) '''
        network_vars = DEFAULT_VARS
        self.residual_keys = network_vars["residual_keys"] # keywords to recognize residual nodes.
        self.attention_key_entry = network_vars["attention_key_entry"] # keywords to recognize entry attention nodes.
        self.attention_key_exit = network_vars["attention_key_exit"]  # keywords to recognize exit attention nodes.
        self.stage = network_vars["stage"]

    def print_(self):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("All merging node info \n")
        utils.print_with_linespace(self.profile)

        print("Merging node-key map \n")
        print(self.merge_key_map,"\n")

        print("Sepcial nodes \n")
        print(self.special_nodes, "\n")

        print("Nodes to compute pre-forward \n")
        print(self.transform_key, "\n")

        # to inspect entire nodes and inf
        # print(self.net.graph)

    def reset_profile(self):
        self.profile = dict() # All mergeable nodes in the network (this is final profile used)

        self.merging_centroid = []
        self.merge_key_map = dict()
        self.special_nodes = [] # Collections of nodes that are specified in SPECIAL_MODULE  

        self.int_features = dict() # Computed only wrt the necessary merging nodes. Features are "pre-forward" features.
        self.tranforms = dict() # Rebasin transforms computed for alignment.

        self.hooks = [] # All the added hooks  

        self.attention_node_id = []
        self.QK_transforms = dict() # there is no inverse transform for {Q,K} transforms
        self.outer_mh_transforms = dict() # outer MH perms 

        self.EMBEDDING_MODULES = ()
        self.EMBEDDING_MODULES_li = []

    def rule_in_embeddings(self):
        '''Early token embeddings and positional encodings'''
        self.EMBEDDING_MODULES = (nn.Embedding,)
        self.EMBEDDING_MODULES_li = ["Embedding"] 

    def extract_profile(self):
        '''
        Get merging nodes and explicitize implicit skip connections
        ''' 
        self.node_index = 1
        if self.fx == torch_FX:
            traced = self.fx.symbolic_trace(self.net)
        else:
            # Huggingface Transformers' symbolic_trace args 
            traced = self.fx.symbolic_trace(self.net,
                                            input_names=["input_ids", "attention_mask"])
        # Print the generated code
        for node in traced.graph.nodes:
            if (node.op == "call_module") or (node.op in ["get_attr"]):
                node_name = node.target
                # print(node_name)
                if node.op in ["get_attr"]:
                    if "weight" not in node_name and "bias" not in node_name:
                        # print(node_name)
                        continue
                        # attribute = getattr(traced, node_name)
                        # if not isinstance(attribute, PARAMETRIC_MODULE_ATT + self.EMBEDDING_MODULES): continue # Not learnable
                    if "bias" in node_name: continue
                    else: node_name = node_name.rsplit('.', 1)[0]

                # module_type = dict(traced.named_modules())[node_name]
                module_type = utils.get_module(self.net, node_name)
                # print(utils.get_module(self.net, node_name).__class__.__name__)
                if isinstance(module_type, PARAMETRIC_MODULE + self.EMBEDDING_MODULES):
                    self.profile[node_name] = {
                                            "merge_key": self.node_index ,
                                            "module_type": module_type.__class__.__name__,
                                            "special_node": True if isinstance(module_type, SPECIAL_MODULE) or (self.residual_keys in node_name) else False,
                                            "transform_id": None, # node of inverse, forward transforms that are composed during merging. 
                                            "attention": (True if (self.attention_key_exit in node_name) and (not self.attention_key_entry in node_name) else False,
                                                        True if self.attention_key_entry in node_name else False)
                                            }   
                    if self.attention_key_exit in node_name:
                        self.attention_node_id.append(self.node_index)

                    if isinstance(module_type, SPECIAL_MODULE): self.special_nodes.append(node_name)
                    self.merge_key_map[self.node_index] = node_name
                    self.node_index += 1

                    if isinstance(module_type, PARAMETRIC_MERGING_MODULE):
                        self.merging_centroid.append(node_name)
            else:
                pass
        # Node loopup for easy access
        self.node_lookup = {node.name: node for node in traced.graph.nodes}
        
        # Assign transform ids for rebasing candidate network by searching the previous and next parametric nodes. 
        self.assign_transform_in_sequence(traced)

    def assign_transform_in_sequence(self, traced):
        ''' 
        Assign transform id relying on the order of graph nodes 
        (e.g., in a correct graph, the eariler nodes are the nodes that are applied first during inference than later nodes).
        '''
        # TODO: Currently, it only supports permutation based matching
         
        all_nodes = list(traced.graph.nodes)
        end_index = 0
        for node_name in self.profile:
            self.profile[node_name]["transform_id"] = [None, None]
            node_id = node_name.replace('.', '_')
            # TODO: replace next to --> graph._node_map
            node_index = next((i for i, node in enumerate(all_nodes) if ((node_id in node.name) 
                                                                        and "bias" not in node.name)), None)
            assert node_index is not None
            end_index = node_index

            if not self.profile[node_name]["special_node"]: #self.profile[node_name]["module_type"] != "Identity":
                # Search prev nodes
                if self.profile[node_name]["module_type"] in (PARAMETRIC_NORMS+self.EMBEDDING_MODULES_li):
                    # TODO this is where self.permutation_based plays
                    # Normlaization nodes have no inverse transform merging (unmerging transforms)
                    pass
                else: 
                    mk, transform_id = self.pre_merging_node_search(all_nodes, end_index, "all")
                    self.profile[node_name]["transform_id"][0] = transform_id[1]

                # Scan post nodes to check the presence of the closest merging centroid
                mk, _ = self.post_merging_node_search(all_nodes, end_index, "centroid")
                self.profile[node_name]["transform_id"][1] = mk

            # Residual nodes
            else:
                if self.profile[node_name]["module_type"] in (PARAMETRIC_NORMS+self.EMBEDDING_MODULES_li):
                    # TODO this is where self.permutation_based plays
                    # in case skip connections include normlaization nodes
                    pass
                else:
                    # Starting index to search is the node whose output is the input to this residual node
                    start_idx_res = next((i for i, node in enumerate(all_nodes) 
                                        if all_nodes[node_index].args[0].name == node.name), None)
                    # Scan prev nodes (first merging node after the starting index)
                    _, transform_id = self.pre_merging_node_search(all_nodes, start_idx_res, "centroid")
                    self.profile[node_name]["transform_id"][0] = transform_id[1]

                # Scan post nodes (first merging node after the starting index)
                _, transform_id = self.pre_merging_node_search(all_nodes, end_index, "centroid")
                self.profile[node_name]["transform_id"][1] = transform_id[1]
          
    def pre_merging_node_search(self, all_nodes, from_idx, cond = "centroid"):
        '''
        Finds first parametric node (either all or norm-exclusive) by checking previous nodes
        from the node of "from_idx
        "'''
        search_condition = self.merging_centroid if cond =="centroid" else self.profile
        for pre_node in reversed(all_nodes[:from_idx]):
            if (pre_node.op == "call_module") or (pre_node.op in ["get_attr"]):
                if ".weight" in pre_node.target:
                    pre_node_key = ".".join(pre_node.target.split(".")[:-1])
                else: pre_node_key = pre_node.target
                if pre_node_key in search_condition and not self.profile[pre_node_key]["special_node"]:
                    return self.profile[pre_node_key]["merge_key"],\
                        self.profile[pre_node_key]["transform_id"]
        return None, [None, None]
    
    def post_merging_node_search(self, all_nodes, from_idx, cond = "centroid"):
        '''
        Finds first parametric node (either all or norm-exclusive) by checking next nodes
        from the node of "from_idx
        "'''
        search_condition = self.merging_centroid if cond =="centroid" else self.profile
        for post_node in all_nodes[from_idx+1:]:
            if (post_node.op == "call_module") or (post_node.op in ["get_attr"]):
                if ".weight" in post_node.target:
                    post_node_key = ".".join(post_node.target.split(".")[:-1])
                else: post_node_key = post_node.target
                if post_node_key in search_condition and not self.profile[post_node_key]["special_node"]:
                    return self.profile[post_node_key]["merge_key"],\
                        self.profile[post_node_key]["transform_id"]
        return None, [None, None]
    
    def get_transform_key(self):
        '''Get keys for intermediate features'''
        self.transform_key = []
        for key in self.profile:
            v, u = self.profile[key]["transform_id"]
            if (v not in self.transform_key) and (v is not None):
                self.transform_key.append(v)
            if u not in self.transform_key and (u is not None):
                self.transform_key.append(u)
        self.transform_key.sort()
        
    def add_hooks(self):
        """ Add new hooks. """
        # Intermediate feature extraction 
        # best to be called when computing intermediate features
        self.clear_hooks()
        def pre_hook(m, input, id):
            self.int_features[id] = input[0].detach().to(self.device) # TODO: consider managing feature shape here
            return None
        for key in self.transform_key: # TODO
            # print(key)
            m = utils.get_module(self.net, self.merge_key_map[key])
            self.hooks.append(m.register_forward_pre_hook(lambda module, 
                                                          input, 
                                                          ln=key: 
                                                          pre_hook(module, input, ln)))
    def clear_hooks(self):
        """ Clear graph hooks. """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_features(self):
        """ Clear the dictionary of the intermediate features. """
        self.int_features.clear()

    def get_intermediate_features(self):
        """ Get intermediate features """
        return self.int_features.copy()
    
    def rule_out_identity_skip(self):
        '''Remove all transform ids in skipconnection to rule out skipconnection from merging'''
        # TODO delete
        skip_list_key = [] 
        for name in self.special_nodes:
            u, v = self.profile[name]["transform_id"]
            if u is not None:
                if u not in skip_list_key:
                    skip_list_key.append(u)
            if v is not None:
                if v not in skip_list_key:
                    skip_list_key.append(v)
        self.remove_transforms(skip_list_key)

    def remove_transforms(self, ids = []):
        ''' Remove the specified transforms from all nodes to rule out them from merging '''
        for key in self.profile:
            u, v = self.profile[key]["transform_id"]
            self.profile[key]["transform_id"] = (None if u in ids else u,
                                                 None if v in ids else v )
    def transform_resperm_precomp(self, account_none = True):
        '''
        Creates transform ids for precomposition of residual permutations into its branching-out root nodes, 
        instead of making precomposition into identity residuals. 

        That is, there is no increase in the number of parameters after merging around at skip-connections.   
        # TODO Need to check validity
        '''
        for key in reversed(self.profile):
            # find residual connections in sequence from the end 
            if self.profile[key]["special_node"] and self.profile[key]["module_type"] == "Identity":
                which_key, replace_with = self.profile[key]["transform_id"]
                if not account_none and (replace_with == None):
                    continue
                if which_key == None:
                    # if self.profile[key]["module_type"] in PARAMETRIC_NORMS:
                    #     self.profile[key]["transform_id"] = (None, None)
                    continue
                # node_id = key.replace('.', '_')
                # node_index = next((i for i, node in enumerate(all_nodes) if ((node_id in node.name) 
                #                                                             and "bias" not in node.name)), None)
                # start_idx_res = next((i for i, node in enumerate(all_nodes) 
                #                       if all_nodes[node_index].args[0].name == node.name), None)
                
                # for pre_node in reversed(all_nodes[:start_idx_res]):
                #     if (pre_node.op == "call_module") or (pre_node.op in ["get_attr"]):
                #         if ".weight" in pre_node.target:
                #             pre_node_key = ".".join(pre_node.target.split(".")[:-1])
                #         else: pre_node_key = pre_node.target
                #         if pre_node_key in self.profile:
                #             u, v = self.profile[pre_node_key]["transform_id"]
                #             if u == which_key:
                #                 u = replace_with
                #             if v == which_key:
                #                 v = replace_with
                #             self.profile[pre_node_key]["transform_id"] = (u, v)
                # self.profile[key]["transform_id"] = (None, None)

                # TODO Very first LN1 can still have permutation (double check skip connection perm simplication) 
                for prev_key in reversed(range(1,self.profile[key]["merge_key"])):
                    u, v = self.profile[self.merge_key_map[prev_key]]["transform_id"]
                    if u == which_key:
                        u = replace_with
                    if v == which_key:
                        v = replace_with
                    self.profile[self.merge_key_map[prev_key]]["transform_id"] = (u, v)
                self.profile[key]["transform_id"] = (None, None)

    def create_profile(self):
        self.reset_profile()

        if self.embeddings:
            self.rule_in_embeddings()

        self.extract_profile()

        if self.prec_residuals:
            self.transform_resperm_precomp()

        self.get_transform_key()

        # raise NotImplementedError("") # TODO remove later on
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("All merging node info \n")
        utils.print_with_linespace(self.profile)
        # self.print_()
        return self
    

##########################################################
##########################################################

class NNProfileGeneral(NNProfileBase):
    '''
    Equiv. NNProfileBase with create_profile() being called at ininitialization
    '''
     # TODO implement ** 
    def __init__(self, 
                 model,
                 network,
                 device = 'cpu',
                 permutation_based = True,

                 embeddings = False,
                 prec_residuals = False
                 ):
        super(NNProfileGeneral, self).__init__(model, network, device, permutation_based, embeddings, prec_residuals)
        
        ''' Implementation-specific vars (default) '''
        network_vars = DEFAULT_VARS
        self.network = network
        self.residual_keys = network_vars["residual_keys"] # keywords to recognize residual nodes.
        self.attention_key_entry = network_vars["attention_key_entry"] # keywords to recognize entry attention nodes.
        self.attention_key_exit = network_vars["attention_key_exit"]  # keywords to recognize exit attention nodes.
        self.stage = network_vars["stage"]

        self.create_profile()


class NNProfileResnet(NNProfileBase):
    '''
    Resnet profile
    '''
     # TODO implement ** 
    def __init__(self, 
                 model,
                 network,
                 device = 'cpu',
                 permutation_based = True,

                 embeddings = False,
                 prec_residuals = False
                 ):
        super(NNProfileResnet, self).__init__(model, network, device, permutation_based, embeddings, prec_residuals)
        
        ''' Implementation-specific vars (default) '''
        network_vars = RESNET_VARS
        self.network = network
        self.residual_keys = network_vars["residual_keys"] # keywords to recognize residual nodes.
        self.attention_key_entry = network_vars["attention_key_entry"] # keywords to recognize entry attention nodes.
        self.attention_key_exit = network_vars["attention_key_exit"]  # keywords to recognize exit attention nodes.
        self.stage = network_vars["stage"]
        self.create_profile()






if __name__ == "__main__":
    from merges.networks import lenet, resnet 

    n = resnet.ResNet18_cifar(input_dim = 3, num_classes=10, zero_init = False, multiplier = 1)
    m = NNProfileResnet(n,    
                        network = "resnet",
                        permutation_based = True,
                        prec_residuals = True

                     )
    m.add_hooks()

    x = torch.randn(2, 3, 32, 32)
    y = m.net(x)

    xx = m.get_intermediate_features()
    i = 0
    for key in xx:
        i += 1
        print(key)
        print(xx[key].shape)
    print(i)