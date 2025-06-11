'''
Test script for merging.
Consider putting them together into a new wrapper class (as a complete pipeline)
'''
import os
import argparse
import torch
import numpy as np
import logging

from tqdm import tqdm
from merging.merges.merge import ModelMerger
from merging.merges.dataset_merge.dataload_ import Load_dataset
from merging.merges.trainings import trainings, validation
from merging.merges import utils
from merging.merges.networks import lenet, resnet

'''Basic test'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./merges/Logs_/logs_', 
                        help="path to save all the products from each trainging")
    parser.add_argument("--id_", type=int, default=1002, help="run id")
    parser.add_argument("--data_path", type=str, default='/data/home/mkim332/ts_project1/datas/sImages',  # TODO: Remove it later on
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


    def load_dataset(mt = True):
        # No explicit validation data are available
        load_, (L, C, K) = Load_dataset(
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
                    "train_test": False,
                    "fix_split_labels": mt})
        if config.channel_in != C:
            config.channel_in = C
        if config.class_num != K: # TODO: FOR NOW WE SET 5:5 SUBSETS FOR MULTITASK SETTING (only the known domain scenario)
            config.class_num = K
        training_data_A, training_data_B = load_.__get_training_loader__()
        testing_data_A, testing_data_B = load_.__get_testing_loader__()
        validation_data_A, validation_data_B = load_.__get_val_loader__()
        return (training_data_A, testing_data_A, validation_data_A), (training_data_B, testing_data_B, validation_data_B)
    

    data_A, data_B = load_dataset(True)

    # model_A = lenet.LeNet(input_dim = config.channel_in, output_dim = config.class_num, 
    #                 dataset="CIFAR-10", batchnorm = True).to(device)

    # model_B = lenet.LeNet(input_dim = config.channel_in, output_dim = config.class_num, 
    #                 dataset="CIFAR-10", batchnorm = True).to(device)
    
    model_A = resnet.ResNet18_cifar(input_dim = config.channel_in, num_classes=config.class_num, zero_init = False, multiplier = 2).to(device)
    model_B = resnet.ResNet18_cifar(input_dim = config.channel_in, num_classes=config.class_num, zero_init = False, multiplier = 2).to(device)

    if config.equalize_init:
        logger.info("Same initialization")
        model_A, model_B = utils.equalize_init(ref = model_A,
                                    target = model_B)

    # Train quickly here
    logger.info("Candidate model A is being trained ...")
    candidate_model_A = trainings.basic(model_A,
                                config, 
                                utils.merge_data_iter(data_A[0]),
                                utils.merge_data_iter(data_A[1]),
                                device,
                                logger,
                                model_id = "Model A",
                                save_cp= False)
    logger.info("Candidate model B is being trained ...")
    candidate_model_B = trainings.basic(model_B,
                                config, 
                                utils.merge_data_iter(data_B[0]),
                                utils.merge_data_iter(data_B[1]),
                                device,
                                logger,
                                model_id = "Model B",
                                save_cp= False)
    
    # Testing net
    acc = validation.vali(candidate_model_A, utils.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy of the candidate network A on dataset A: {np.abs(acc):.6f}%" )

    acc = validation.vali(candidate_model_A, utils.merge_data_iter(data_B[1]), device)
    logger.info(f"Testing accuracy of the candidate network A on dataset B: {np.abs(acc):.6f}% \n" )

    acc = validation.vali(candidate_model_B, utils.merge_data_iter(data_A[1]), device)
    logger.info(f"Testing accuracy of the candidate network B on dataset A: {np.abs(acc):.6f}%" )

    acc = validation.vali(candidate_model_B, utils.merge_data_iter(data_B[1]), device)
    logger.info(f"Testing accuracy of the candidate network B on dataset B: {np.abs(acc):.6f}% \n" )


    # merge 
    candidate_model_A.eval()
    candidate_model_B.eval()

    merge_dataloader = utils.merge_dataloader(data_A[0], data_B[0],
                                            batch_size = config.batch,
                                            subset_p = 0.4)

    merger = ModelMerger(candidate_model_A, candidate_model_B,
                        network = config.network,
                        dataset = merge_dataloader,
                        device = device,
                        batch_for_sim = config.batch_for_sim,
                        
                        align = True,
                        align_method="Permutation",
                        align_type="AM",
                        permutation_based= True,
                        embeddings= config.embeddings,
                        prec_residuals = config.prec_residuals,
                        qk_perm= config.qk_perm,
                        merge_method = "average"
                        
                        )
    merged_model, rebasined_model_A, rebasined_model_B = merger.merge()


    # Merge test
    recompute_batchnorm_stat = False
    if recompute_batchnorm_stat:
        logger.info("Recomputing batchnorm statistics is made")
        full_merged_data = utils.merge_dataloader(data_A[0], data_B[0],
                                    batch_size = config.batch,
                                    subset_p = 1.)
        merged_model = validation.recompute_batchnorm_stat(merged_model, utils.merge_data_iter(full_merged_data), 
                                                                device, repeat = 2)
        
        # rebasined_model_A = validation.recompute_batchnorm_stat(rebasined_model_A, utils.merge_data_iter(data_A[0]), 
        #                                                         device, repeat = 2)
                                                                
        # rebasined_model_B = validation.recompute_batchnorm_stat(rebasined_model_B, utils.merge_data_iter(data_B[0]), 
        #                                                         device, repeat = 2)
        
    logger.info("Validation is being made ... \n")
    for name_, model in zip(["rebasined A", "rebasined B", "merged net (wrt. A)"], [rebasined_model_A, rebasined_model_B, merged_model]):
        acc = validation.vali(model, utils.merge_data_iter(data_A[1]), device)
        logger.info(f"Testing accuracy of {name_} on dataset A: {np.abs(acc):.6f}%" )

    logger.info("\n")

    for name_, model in zip(["rebasined A", "rebasined B", "merged net (wrt. A)"], [rebasined_model_A, rebasined_model_B, merged_model]):
        acc = validation.vali(model, utils.merge_data_iter(data_B[1]), device)
        logger.info(f"Testing accuracy of {name_} on dataset B: {np.abs(acc):.6f}%" )

    logger.info("Validation is done") 

    # # Quick sanity check

    logger.info("") 

    merging_f, merging_f_B, inspection_ = merger.get_data() 

    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()

    plt.rcParams['agg.path.chunksize'] = 1000
    sns.set(style='ticks', font_scale=1.2)
    plt.rcParams['figure.figsize'] = 12,8
    for i, key in enumerate(tqdm(inspection_, desc = "Visualization of inspection_: ")):
        # logger.info(f"{key}: {inspection_[key].get().shape}")
        for name_, mf, mf_B in zip(["forward", "inverse"], [merging_f[key]["forward"], merging_f[key]["inverse"]],
                                [merging_f_B[key]["forward"], merging_f_B[key]["inverse"]]):
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600)
            axes = fig.subplots()        
            im = axes.imshow(torch.cat((mf,mf_B), dim = 1).detach().cpu().numpy(), aspect='auto', cmap='viridis')
            cbar = plt.colorbar(im)
            cbar.set_label('Weight', fontweight='bold')

            plt.xlabel('Input Dimension')
            plt.ylabel('Output Dimension')

            plt.savefig(os.path.join(config.plots_save_path, f"layer{key}_{name_}_" + ".png" ), bbox_inches='tight') 
            plt.clf()   
            plt.close(fig)

        fig = plt.figure(figsize=(12, 8), 
        dpi = 600)
        axes = fig.subplots()        
        im = axes.imshow(inspection_[key].get().detach().cpu().numpy(), aspect='auto', cmap='viridis')
        cbar = plt.colorbar(im)
        cbar.set_label('Weight', fontweight='bold')

        plt.xlabel('Input Dimension')
        plt.ylabel('Output Dimension')

        plt.savefig(os.path.join(config.plots_save_path, f"metric_layer{key}_" + ".png" ), bbox_inches='tight') 
        plt.clf()   
        plt.close(fig)
