import os
import argparse
import torch
import numpy as np
import logging
from copy import deepcopy
from tqdm import tqdm
import merging as mg
from NNConnectivitiy.NNC_analysis import plot_1d
from NNConnectivitiy.LS_plane import *
# from merges.merge import ModelMerger
# from merges.dataset_merge.dataload_ import Load_dataset
# from merges.trainings import trainings, validation
# from merges import utils
# from merges.networks import lenet, resnet

'''Basic test'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./merges/Logs_/logs_', 
                        help="path to save all the products from each trainging")
    parser.add_argument("--id_", type=int, default=1002, help="run id")
    parser.add_argument("--data_path", type=str, default="",  
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
    
    data_A, data_B, data_C = load_dataset(False)
    
    # Get dataloader of {train & test}
    train_test = mg.merge_dataloader(data_A[0] if data_C is None else data_C, data_A[1],
                                            batch_size = config.batch)
    
    models = []
    for i in range(2):
        network = {"lenet": mg.LeNet(input_dim = config.channel_in, output_dim = config.class_num, dataset=config.dataset, batchnorm = False),
               "resnet": mg.ResNet18_cifar(input_dim = config.channel_in, num_classes=config.class_num, zero_init = False, multiplier = 2)
               }[config.network]
        models.append(network)
        
    model_A = models[0]
    model_B = models[1]
    
    if config.equalize_init:
        logger.info("Same initialization")
        model_A, model_B = mg.equalize_init(ref = model_A, target = model_B)
    init_model = deepcopy(model_A)
    # Check loadability 
    model_A, required_training_A = mg.load_model(model_A, config.model_save_path, model_name= "Model A", g_logger=logger)
    model_A = model_A.to(device)
    model_B, required_training_B = mg.load_model(model_B, config.model_save_path, model_name= "Model B", g_logger=logger)
    model_B = model_B.to(device)
    # Training
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
    
    if (config.train_mode == 1) or (config.train_mode == 2):
        model_C = deepcopy(model_A)
        model_C, required_training_C = mg.load_model(model_C, config.model_save_path, model_name= "Model C", g_logger=logger)
        model_C = model_C.to(device)
        if required_training_C:
            logger.info("Candidate model C is being trained on TESTING set...")
            model_C = mg.basic(model_C,
                            config, 
                            mg.merge_data_iter(data_A[1]),
                            mg.merge_data_iter(data_A[1]),
                            device,
                            logger,
                            model_id = "Model C",
                            save_cp= True)
        else: print("Trained Model C has been loaded")
    else:
        model_C = None
            
    if required_training_B:
        logger.info("Candidate model B is being trained on training & TESTING sets...")
        model_B = mg.basic(model_B,
                        config, 
                        mg.merge_data_iter(train_test),
                        mg.merge_data_iter(data_A[1]),
                        device,
                        logger,
                        model_id = "Model B",
                        save_cp= True)
    else: print("Trained Model B has been loaded")
    
    if config.train_mode == 2:
        model_D = deepcopy(init_model)
        model_D = model_D.to(device)
        logger.info("Candidate model C is being trained on TESTING set...")
        model_D = mg.basic(model_D,
                        config, 
                        mg.merge_data_iter(data_A[1]),
                        mg.merge_data_iter(data_A[1]),
                        device,
                        logger,
                        model_id = "Model D",
                        save_cp= True)
    else: model_D = None
    # Testing net
    acc = mg.vali(model_A, mg.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy of the network A: {np.abs(acc):.6f}%" )
    acc = mg.vali(model_A, mg.merge_data_iter(data_A[0] if data_C is None else data_C ), device)
    logger.info(f"Training accuracy of the network A: {np.abs(acc):.6f}%" )
    
    acc = mg.vali(model_B, mg.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy of the network B: {np.abs(acc):.6f}% \n" )
    acc = mg.vali(model_B, mg.merge_data_iter(data_A[0] if data_C is None else data_C), device)
    logger.info(f"Training accuracy of the network B: {np.abs(acc):.6f}% \n" )
    if config.train_mode == 1 or config.train_mode == 2:
        acc = mg.vali(model_C, mg.merge_data_iter(data_A[1]), device)
        logger.info(f"Testing accuracy of the network C: {np.abs(acc):.6f}%" )
        acc = mg.vali(model_C, mg.merge_data_iter(data_A[0] if data_C is None else data_C ), device)
        logger.info(f"Training accuracy of the network C: {np.abs(acc):.6f}%" )
        if config.train_mode == 2:
            acc = mg.vali(model_D, mg.merge_data_iter(data_A[1]), device)
            logger.info(f"Testing accuracy of the network D: {np.abs(acc):.6f}%" )
            acc = mg.vali(model_D, mg.merge_data_iter(data_A[0] if data_C is None else data_C ), device)
            logger.info(f"Training accuracy of the network D: {np.abs(acc):.6f}%" )
    # plot_1d(model_A, model_B, data_A[1], num_alpha=21, log_plot_path=config.plots_save_path,
    #         device = device, title= "test")
    
    # plot_1d(model_A, model_B, data_A[0], num_alpha=21, log_plot_path=config.plots_save_path,
    #         device = device, title = "train")
    
    construct_loss_surface(config,
                       model_A, model_B, init_model if model_C is None else model_C,
                       data_A[1],
                       device = device,
                       
                       # plane arguments
                       grid_dim =21, # plane is of grid by grid (determines resolution)
                       margin = [0.2, 0.2, 0.2, 0.2], # left right bottom top
                       num_from_path = 20, # num. points in linear interpolation.  

                       save_results = True,
                       save_name = "test",
                       Model_D = model_D)
    plot_loss_surface(config, save_name = "test")
    
    
    construct_loss_surface(config,
                       model_A, model_B, init_model if model_C is None else model_C,
                       data_A[1],
                       device = device,
                       
                       # plane arguments
                       grid_dim =21, # plane is of grid by grid (determines resolution)
                       margin = [0.2, 0.2, 0.2, 0.2], # left right bottom top
                       num_from_path = 20, # num. points in linear interpolation.  

                       save_results = True,
                       save_name = "train",
                       Model_D = model_D)
    plot_loss_surface(config, save_name = "train")