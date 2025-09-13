'''
Main run file for Hypernetwork
'''
import os
import argparse
import torch
import numpy as np
import logging
from copy import deepcopy
from tqdm import tqdm

import merging as mg
import Hypernetwork
from NNConnectivitiy.NNC_analysis import plot_1d, Interpolation

def model_size(m, model_name, ):
        logger.info(f"Model: {model_name}")
        total_param = 0
        for name, param in m.named_parameters():
            num_params = param.numel()
            total_param += num_params
            logger.info(f"{name}: {num_params} parameters")
        
        logger.info(f"Total parameters in {model_name}: {total_param}")
        logger.info("")
        return total_param

def model_inf(model, logger, model_name):
    
    model_name = [model_name]
    m = [model]
    total_p = 0
    for i, name in enumerate(model_name):
        param_ = model_size(m[i], name) if m[i] is not None else 0
        total_p += param_
    logger.info(f"Total trainable parameters in {model_name[0]}: {total_p}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./merges/Logs_/logs_', 
                        help="path to save all the products from each trainging")
    parser.add_argument("--id_", type=int, default=1002, help="run id")
    parser.add_argument("--data_path", type=str, default='/data/home/mkim332/ts_project1/datas/sImages',  
                        help="path to grab data")
    parser.add_argument("--dataset", type=str, default="CIFAR-10", choices=["MNIST", "CIFAR-10", 
                                                                             "CIFAR-100", "SVHN", 
                                                                             "STL10", "CelebA"])
    # Save path
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--plots_save_path', type=str, default='plots')
    parser.add_argument('--his_save_path', type=str, default='hist')

    # Training params
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--gpu_dev", type=str, default="5")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--epochs_hn", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--batch_testing", type=int, default=256)
    parser.add_argument("--data_transform_set", type=list, default=[False, False, False, False], 
                        help = "[random_crop, autoaug, random_crop_pase, resize]")

    parser.add_argument("--train_mode", type=int, default=0, help = "0: train vs {train & test}, ")
    
    
    # optimizer
    parser.add_argument("--opti", type=str, default="sgd")
    parser.add_argument("--lr_", type=float, default=8e-3, help= "Peak learning rate")
    parser.add_argument("--scheduler", type=int, default=0, help= "Whether to use optimizer scheduler for lr and wd")
    parser.add_argument("--wd", type=str, default= "None", choices=["L1_main", "L2_main", "L1_subspace", "L2_subspace", "None"])
    parser.add_argument("--wd_rate", type=float, default=1e-3, help= "Peak learning rate")
    parser.add_argument("--grokking", type=int, default=0)

    ### Scheduler params
    parser.add_argument("--warm_up", type=float, default=0.1, help="portion of warm up given number of epoches, e.g., 20 percent by defualt")
    parser.add_argument("--start_lr", type=float, default=1e-4, help="starting learning rate")
    parser.add_argument("--ref_lr", type=float, default=3e-3, help= "Peak learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-3, help = "final learning rate")
    parser.add_argument("--start_wd", type=float, default=0.01, help = "starting weight decay")
    parser.add_argument("--final_wd", type=float, default=0.0001, help = "fianl weight decay")

    parser.add_argument("--channel_in", type=int, default=3)
    parser.add_argument("--class_num", type=int, default=10)


    parser.add_argument("--equalize_init", type=int, default=1)
    parser.add_argument("--batch_for_sim", type=int, default=32)
    
    # Merge params
    parser.add_argument("--embeddings", type=int, default=0)
    parser.add_argument("--prec_residuals", type=int, default=0)
    parser.add_argument("--qk_perm", type=int, default=0)
    
    parser.add_argument("--network", type=str, default="resnet")
    parser.add_argument("--init_hyper", type=int, default=0)
    parser.add_argument("--hypernet_training", type=str, default="reconstruction", choices = {"reconstruction", "prediction"})
    parser.add_argument("--lowrank", type=int, default=0)
    parser.add_argument("--decomposition", type=int, default=0)
    parser.add_argument("--node_direction", type=str, default="W")
    parser.add_argument("--hypermatching", type=int, default=0)
    parser.add_argument("--gradient_matching", type=int, default=0)
    parser.add_argument("--intrinsic_training", type=int, default=0)
    parser.add_argument("--multitask", type=int, default=1)
    config = parser.parse_args()
    log_path = config.log_path + config.dataset + "_" + f"{config.id_}" 

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    os.environ["log_loc"] = f"{log_path}"
    root_dir = os.getcwd() 
    logging.basicConfig(filename=os.path.join(root_dir, f'{log_path}/log_all.txt'), level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('In main')

    logger.info(f"********************* Setup the hyperparams *********************")
    config.model_save_path = os.path.join(log_path,"checkpoints") 
    config.plots_save_path = os.path.join(log_path,"plots") 
    config.his_save_path = os.path.join(log_path,"hist") 
    ## Training and testing logging path and logger
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.plots_save_path):
        os.makedirs(config.plots_save_path)
    if not os.path.exists(config.his_save_path):
        os.makedirs(config.his_save_path)
    
    if config.dataset == "MNIST":
        config.image_size = 28
    elif config.dataset == "CIFAR-10":
        config.image_size = 32
    args = vars(config)

    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(f'cuda:{config.gpu_dev}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"GPU (device: {device}) used" if torch.cuda.is_available() else 'cpu used')


    def load_dataset(mt = False):
        # No explicit validation data are available
        load_, (L, C, K) = mg.Load_dataset(
                    {"data_path": config.data_path,
                    "sub_dataset": config.dataset,
                    "permute": False,
                    "sqw": False,
                    "batch_training": config.batch ,
                    "batch_testing": config.batch_testing ,
                    "rc": bool(config.data_transform_set[0]),
                    "aa": bool(config.data_transform_set[1]),
                    "rcp": bool(config.data_transform_set[2]),
                    "rs": bool(config.data_transform_set[3]),
                    "multitasking": mt,
                    "train_test": True,
                    "fix_split_labels": mt})
        if config.channel_in != C:
            config.channel_in = C
        if config.class_num != K: # TODO: FOR NOW WE SET 5:5 SUBSETS FOR MULTITASK SETTING (only the known domain scenario)
            config.class_num = K
        training_data_A, training_data_B = load_.__get_training_loader__()
        testing_data_A, testing_data_B = load_.__get_testing_loader__()
        validation_data_A, validation_data_B = load_.__get_val_loader__()
        training_loader_balanced, _ = load_.__get_balanced_training_loader__()
        
        return (training_data_A, testing_data_A, validation_data_A), (training_data_B, testing_data_B, validation_data_B), training_loader_balanced
    
    data_A, data_B, data_C = load_dataset(bool(config.multitask))
    
    
    models = []
    for i in range(3 if config.init_hyper else 1):
        network = {"lenet": mg.LeNet(input_dim = config.channel_in, output_dim = config.class_num, dataset=config.dataset, batchnorm = False),
                   "mlp": mg.MLP(input_dim = config.channel_in, output_dim = config.class_num, dataset=config.dataset),
               "resnet": mg.ResNet18_cifar(input_dim = config.channel_in, num_classes=config.class_num, zero_init = False, multiplier = 2),
               "mlpmixer": mg.MLPMixer(input_dim=config.channel_in, image_size=config.image_size,num_classes=config.class_num,num_layers=3)
               }[config.network]
        models.append(network)
        
    model_A = models[0]
    model_B = models[1]
    model_inf(model_A, logger, model_name= "Model_A")
    if config.equalize_init:
        # Don't equalize
        logger.info("Same initialization")
        model_A, model_B = mg.equalize_init(ref = model_A, target = model_B)
    if config.init_hyper:
        init_model = models[2].to(device)
        print("Init hypernet")
    else: init_model = None
    # Check loadability 
    model_A, required_training_A = mg.load_model(model_A, config.model_save_path, model_name= "Model A", g_logger=logger)
    model_A = model_A.to(device)
    model_B, required_training_B = mg.load_model(model_B, config.model_save_path, model_name= "Model B", g_logger=logger)
    model_B = model_B.to(device)
    
    # Training the target nets
    if required_training_A:
        logger.info("Candidate model A is being trained on TRAINING set...")
        model_A = mg.basic(model_A,
                        config, 
                        mg.merge_data_iter(data_A[0]),
                        mg.merge_data_iter(data_A[1]),
                        device,
                        logger,
                        model_id = "Model A",
                        save_cp= True)
    else: print("Trained Model A has been loaded")
    
    if required_training_B:
        logger.info("Candidate model B is being trained on training & TESTING sets...")
        model_B = mg.basic(model_B,
                        config, 
                        mg.merge_data_iter(data_B[0] if config.multitask else data_A[0]),
                        mg.merge_data_iter(data_B[1] if config.multitask else data_A[1]),
                        device,
                        logger,
                        model_id = "Model B",
                        save_cp= True)
    else: print("Trained Model B has been loaded")
    
    # Testing the target nets
    acc = mg.vali(model_A, mg.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy (dataset A) of the network A: {np.abs(acc):.6f}%" )
    acc = mg.vali(model_A, mg.merge_data_iter(data_B[1] if config.multitask else data_A[1] ), device) # data_B[1] if data_C is None else data_C 
    logger.info(f"Training accuracy (dataset B) of the network A: {np.abs(acc):.6f}%" )
    
    acc = mg.vali(model_B, mg.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy (dataset A) of the network B: {np.abs(acc):.6f}% \n" )
    acc = mg.vali(model_B, mg.merge_data_iter(data_B[1] if config.multitask else data_A[1] ), device) #data_B[1] if data_C is None else data_C 
    logger.info(f"Training accuracy (dataset B) of the network B: {np.abs(acc):.6f}% \n" )
    
    # plot_1d(model_A, model_B, data_A[1], num_alpha=21, log_plot_path=config.plots_save_path,
    #         device = device, title= "data_A")
    
    # plot_1d(model_A, model_B, data_B[1], num_alpha=21, log_plot_path=config.plots_save_path,
    #         device = device, title = "data_B")
    # Hypernetwork.weights_vis(model_A, model_B, dir = config.plots_save_path, title = "original_A_and_B") 
    plot_1d(model_A, model_B, data_A[1], num_alpha=21, log_plot_path=config.plots_save_path,
            device = device, title= "original_data_A", testing_data_B=data_B[1] if config.multitask else data_A[1])
    # if config.init_hyper:
    #     init_model = Interpolation(model_A, model_B, alpha=0.5)
    #     init_model = init_model.to(device)
    #     print("Re-Init hypernet")
        
    base_hypernet = "transformer"
    # Train hypernets
    hypernet_A = Hypernetwork.Framework_HN(model_A, config.model_save_path, name = "hypernet_A", device = device, init_net=init_model,
                                           hypernet_base= base_hypernet,
                                           compression = bool(config.decomposition),
                                           node_direction = "W",
                                           lowrank = bool(config.lowrank),
                                           rank = 24,
                                            type_ = "lowrank",
                                            decomposition = bool(config.decomposition),
                                            learnable_emb= True,
                                            hypermatching=config.hypermatching,
                                            zero_init_emb=False,
                                            intrinsic_training=config.intrinsic_training).to(device)
    hypernet_B = Hypernetwork.Framework_HN(model_B, config.model_save_path, name = "hypernet_B", device = device, init_net=init_model,
                                           hypernet_base= base_hypernet,
                                           compression = bool(config.decomposition),
                                           node_direction = "W",
                                           lowrank = bool(config.lowrank),
                                           rank = 24,
                                            type_ = "lowrank",
                                            decomposition = bool(config.decomposition),
                                            learnable_emb= True,
                                            hypermatching=config.hypermatching,
                                            zero_init_emb=False,
                                            intrinsic_training=config.intrinsic_training).to(device)
    model_inf(hypernet_A, logger, model_name= "Hypernet_A")
    # Pretrained loaded
    if hypernet_A.checkpoint and hypernet_B.checkpoint:
        pass
    else:
        logger.info("****************** Train new hypernets ******************")
        # Equalize the hypernet init
        hypernet_B.init_Hypernet(hypernet_A, type = base_hypernet)
        
        # Hypernetwork.weights_vis(hypernet_A, hypernet_B, dir = config.plots_save_path, title = "is_hypernet_equalized")
        # Training
        if config.hypernet_training == "reconstruction":
            if config.hypermatching: # training "shared" hypernets including the subspace params too --> TODO
                hypernet_A, hypernet_B = Hypernetwork.sim_basic_hn(hypernet_A, hypernet_B, config, 
                                                mg.merge_data_iter(data_A[0]),
                                                mg.merge_data_iter(data_A[1]),
                                                device = device, g_logger = logger, model_id = "Hypernet_A",
                                                emb_vis = False, intrinsic_training=config.intrinsic_training)
            elif config.gradient_matching: # independent training with grad reg
                hypernet_A, hypernet_B = Hypernetwork.sim_reg_hn(hypernet_A, hypernet_B, config, 
                                                mg.merge_data_iter(data_A[0]),
                                                mg.merge_data_iter(data_A[1]),
                                                device = device, g_logger = logger, model_id = "Hypernet_A",
                                                emb_vis = False, intrinsic_training=config.intrinsic_training)
            else: # independent training 
                hypernet_A = Hypernetwork.basic_hn(hypernet_A, config, 
                                                mg.merge_data_iter(data_A[0]),
                                                mg.merge_data_iter(data_A[1]),
                                                device = device, g_logger = logger, model_id = "Hypernet_A",
                                                emb_vis = False, intrinsic_training=config.intrinsic_training)
                
                hypernet_B = Hypernetwork.basic_hn(hypernet_B, config, 
                                                mg.merge_data_iter(data_B[0] if config.multitask else data_A[0]),
                                                mg.merge_data_iter(data_B[1] if config.multitask else data_A[1]),
                                                device = device, g_logger = logger, model_id = "Hypernet_B",
                                                intrinsic_training=config.intrinsic_training)
                
            hypernet_A.save_hypernet(name = "hypernet_A")
            hypernet_B.save_hypernet(name = "hypernet_B")
        elif config.hypernet_training == "prediction": # independent training 
            hypernet_A = Hypernetwork.prediction_train(hypernet_A, config, 
                                            mg.merge_data_iter(data_A[0]),
                                            mg.merge_data_iter(data_A[1]),
                                            device = device, g_logger = logger, model_id = "Hypernet_A",
                                            emb_vis = False,
                                            intrinsic_training=config.intrinsic_training)
    
            hypernet_B = Hypernetwork.prediction_train(hypernet_B, config, 
                                            mg.merge_data_iter(data_B[0] if config.multitask else data_A[0]),
                                            mg.merge_data_iter(data_B[1] if config.multitask else data_A[1]),
                                            device = device, g_logger = logger, model_id = "Hypernet_B",
                                            intrinsic_training=config.intrinsic_training)
        
    # Yield the learned target nets 
    # TODO not sure if this is trainable (if not, need to copy the param state to trainable net)
    
    model_A2 = hypernet_A.marterialize_Implicitnet(deepcopy(model_A))
    model_B2 = hypernet_B.marterialize_Implicitnet(deepcopy(model_B))
        
    # Testing the generated nets
    acc = mg.vali(model_A, mg.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy (dataset A) of the network A - after hypernet : {np.abs(acc):.6f}%" )
    acc = mg.vali(model_A, mg.merge_data_iter(data_B[1] if config.multitask else data_A[1] ), device)
    logger.info(f"Training accuracy (dataset B) of the network A - after hypernet : {np.abs(acc):.6f}%" )
    
    acc = mg.vali(model_B, mg.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy (dataset A) of the network B - after hypernet : {np.abs(acc):.6f}% \n" )
    acc = mg.vali(model_B, mg.merge_data_iter(data_B[1] if config.multitask else data_A[1]), device)
    logger.info(f"Training accuracy (dataset B) of the network B - after hypernet : {np.abs(acc):.6f}% \n" )
    
    # Testing the generated nets
    acc = mg.vali(model_A2, mg.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy (dataset A) of the network A: {np.abs(acc):.6f}%" )
    acc = mg.vali(model_A2, mg.merge_data_iter(data_B[1] if config.multitask else data_A[1] ), device)
    logger.info(f"Training accuracy (dataset B) of the network A: {np.abs(acc):.6f}%" )
    
    acc = mg.vali(model_B2, mg.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy (dataset A) of the network B: {np.abs(acc):.6f}% \n" )
    acc = mg.vali(model_B2, mg.merge_data_iter(data_B[1] if config.multitask else data_A[1]), device)
    logger.info(f"Training accuracy (dataset B) of the network B: {np.abs(acc):.6f}% \n" )
  
    # plot_1d(model_A2, model_B2, data_A[1], num_alpha=21, log_plot_path=config.plots_save_path,
    #         device = device, title= "hn_data_A")
    
    # plot_1d(model_A2, model_B2, data_B[1], num_alpha=21, log_plot_path=config.plots_save_path,
    #         device = device, title = "hn_data_B")
    plot_1d(model_A2, model_B2, data_A[1], num_alpha=21, log_plot_path=config.plots_save_path,
            device = device, title= "hn_data_A", testing_data_B=data_B[1] if config.multitask else data_A[1])
      
    # Hypernetwork.weights_vis(model_A, model_A2, dir = config.plots_save_path, title = "A-A2") # TODO WHY NOT WORKING ?? 
    
    # Hypernetwork.weights_vis(model_A2, model_B2, dir = config.plots_save_path, title = "A-B") # TODO WHY NOT WORKING ?? 
    
    # Hypernetwork.weights_vis(hypernet_A, hypernet_B, dir = config.plots_save_path, title = "hypernet_A-B") # TODO WHY NOT WORKING ?? 
    
    # Hypernetwork.Hypernet_plot_1d(hypernet_A, hypernet_B, model_A, data_A[1], num_alpha=21, log_plot_path=config.plots_save_path,
    #         device = device, title= "hn_data_A")
    
    # Hypernetwork.Hypernet_plot_1d(hypernet_A, hypernet_B, model_A, data_B[1], num_alpha=21, log_plot_path=config.plots_save_path,
    #     device = device, title= "hn_data_B")
    Hypernetwork.Hypernet_plot_1d(hypernet_A, hypernet_B, model_A, data_A[1], num_alpha=21, log_plot_path=config.plots_save_path,
            device = device, title= "hn_data_A", testing_data_B= data_B[1] if config.multitask else data_A[1])
    
    if config.intrinsic_training:
        Hypernetwork.Subspace_plot_1d(hypernet_A, hypernet_B, model_A, data_A[1], num_alpha=21, log_plot_path=config.plots_save_path,
                device = device, title= "hn_data_A_subspace", testing_data_B= data_B[1] if config.multitask else data_A[1])
    # ## Loop 
    Hypernetwork.Hypernet_plot_1d2(hypernet_A, hypernet_B, model_A, data_A[1], num_alpha=21, log_plot_path=config.plots_save_path,
            device = device, title= "hn_data_A_gen_only", testing_data_B= data_B[1] if config.multitask else data_A[1])
    # # Merge hypernets (merging framework)
    # ...
    
    # # Generation from the merged hypernets
    
    # # Testing
    