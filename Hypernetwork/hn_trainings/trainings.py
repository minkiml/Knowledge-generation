import time
import torch
from tqdm import tqdm
from merging.merges.training_utils.logger_ import Value_averager, Logger, ModelSaver_in_merge
from merging.merges.training_utils.optimizer import opt_constructor
from .validation import prediction_vali
import torch.nn.functional as F

def prediction_train(hypernet,
            config, 
            training_data,
            testing_data,
            device,
            g_logger,
            model_id = "",
            save_cp = True,
            emb_vis = False,
            intrinsic_training = False
            ):
    g_logger.info("")
    g_logger.info(f"************************************ Basic training session ({model_id})************************************")
    g_logger.info("")
    
    if not intrinsic_training:
        m = [hypernet]
    else:
        m = [hypernet.hypernet.all_emb]
    ipe = len(training_data)
    optimizer_, lr_scheduler_, wd_scheduler_ = opt_constructor(bool(config.scheduler),
                                                        m,
                                                        lr = config.lr_,

                                                        warm_up = config.epochs_hn* ipe * config.warm_up , #int(self.epochs_hn* ipe * self.warm_up),
                                                        fianl_step = config.epochs_hn* ipe, #int(self.epochs_hn* ipe),
                                                        start_lr = config.start_lr,
                                                        ref_lr = config.ref_lr,
                                                        final_lr = config.final_lr,
                                                        start_wd = config.start_wd,
                                                        final_wd = config.final_wd,
                                                        optimizer= config.opti,
                                                        wd_adam = None # 5e-4
                                                        )
    
    # # Logger construction
    training_log = Logger(config.plots_save_path, 
                        config.his_save_path,
                        model_id,
                        ('%d', 'epoch'),
                        ('%d', 'itr'),
                        ('%.5f', 'loss_CE'),
                        ('%.5f', 'lr'),
                        ('%.5f', 'wd'),

                        ('%.5f', 'train_acc'),
                        ('%.5f', 'val_acc'),
                        )
    if emb_vis:
        hypernet.vis_embeddings(training_log.log_feature_emb)
        hypernet.vis_embeddings_out(training_log.log_feature_emb)
    m_saver= ModelSaver_in_merge(g_logger=g_logger, model_name= model_id, save_cp = save_cp)
    loss_CE = Value_averager()
    obs1 = Value_averager()
    obs2 = Value_averager()

    grads = None

    for epoch in tqdm(range(config.epochs_hn), desc = f"Training {model_id} / {config.dataset}"):
            speed_t = []
            epoch_time = time.time()
            # print(slmc.theta_s.from_theta_g.detach())
            for i, batches in enumerate(zip(*training_data)):
                if len(batches) == 1:
                    x, y = batches[0]
                else:
                    x = torch.cat((batches[0][0],batches[1][0]), dim = 0)
                    y = torch.cat((batches[0][1],batches[1][1]), dim = 0)
                input = x.to(device)######################
                y = y.to(device)
                hypernet.train()
                hypernet.implicitnet_train()
                optimizer_.zero_grad()
                per_itr_time = time.time()

                logits = hypernet.forward_implicitnet(input)
                # print(logits[0:2])
                # print(logits)
                loss = F.cross_entropy(logits, y.detach(), label_smoothing=0.25) / x.shape[0]
                loss.backward() 
                #  ✅ (1) Gradient norm diagnostics
                # print("🔍 Gradient norms per parameter:")
                # for name, param in hypernet.named_parameters():
                #     if param.requires_grad:
                #         if param.grad is None:
                #             print(f"{name}: ❌ No gradient")
                #         else:
                #             print(f"{name}: ✅ grad norm = {param.grad.norm().item():.4e}")
         
                optimizer_.step()

                speed_t.append(time.time() - per_itr_time)

                loss_CE.update(loss.mean().item()) 

                training_log.log_into_csv_(epoch+1,
                                                i,
                                                loss_CE.avg,
                                                obs1.avg, #0.,
                                                obs2.avg) #0.)
              
            # g_logger.info(f"epoch[{epoch+1}/{config.fitting_epoch}], grad (p):{grad1.avg: .6f}")
            g_logger.info(f" epoch[{epoch+1}/{config.epochs_hn}] & speed per epoch: {(time.time() - epoch_time): .5f}, Loss_CE (g -> p):{loss_CE.avg: .4f}")

            acc = prediction_vali(hypernet, testing_data, device)

            m_saver(acc, hypernet, config.model_save_path)
            g_logger.info("")
            g_logger.info("")
    g_logger.info("Training is done .... ")
    g_logger.info("")
    g_logger.info("Evaluation is making .... ")
    hypernet.eval()
    hypernet.implicitnet_train(False)
    hypernet = m_saver.get_best_model(hypernet,config.model_save_path)
    if emb_vis:
        hypernet.vis_embeddings_out(training_log.log_feature_emb, etc = "after")
    return hypernet

def sim_prediction_train(hypernet, hypernet_B,
            config, 
            training_data,
            testing_data,
            
            device,
            g_logger,
            model_id = "",
            save_cp = True,
            emb_vis = False,
            intrinsic_training = False
            ):
    g_logger.info("")
    g_logger.info(f"************************************ Basic training session ({model_id})************************************")
    g_logger.info("")
    
    if not intrinsic_training:
        m = [hypernet]
    else:
        m = [hypernet.hypernet.all_emb]
    ipe = len(training_data[0])
    optimizer_, lr_scheduler_, wd_scheduler_ = opt_constructor(bool(config.scheduler),
                                                        m,
                                                        lr = config.lr_,

                                                        warm_up = config.epochs_hn* ipe * config.warm_up , #int(self.epochs_hn* ipe * self.warm_up),
                                                        fianl_step = config.epochs_hn* ipe, #int(self.epochs_hn* ipe),
                                                        start_lr = config.start_lr,
                                                        ref_lr = config.ref_lr,
                                                        final_lr = config.final_lr,
                                                        start_wd = config.start_wd,
                                                        final_wd = config.final_wd,
                                                        optimizer= config.opti,
                                                        wd_adam = None # 5e-4
                                                        )
    # m2 = [hypernet_B]
    # optimizer_B, lr_scheduler_B, wd_scheduler_B = opt_constructor(bool(config.scheduler),
    #                                                 m2,
    #                                                 lr = config.lr_,

    #                                                 warm_up = config.epochs_hn* ipe * config.warm_up , #int(self.epochs_hn* ipe * self.warm_up),
    #                                                 fianl_step = config.epochs_hn* ipe, #int(self.epochs_hn* ipe),
    #                                                 start_lr = config.start_lr,
    #                                                 ref_lr = config.ref_lr,
    #                                                 final_lr = config.final_lr,
    #                                                 start_wd = config.start_wd,
    #                                                 final_wd = config.final_wd,
    #                                                 optimizer= config.opti,
    #                                                 wd_adam = None # 5e-4
    #                                                 )
    # # Logger construction
    training_log = Logger(config.plots_save_path, 
                        config.his_save_path,
                        model_id,
                        ('%d', 'epoch'),
                        ('%d', 'itr'),
                        ('%.5f', 'loss_CE'),
                        ('%.5f', 'lr'),
                        ('%.5f', 'wd'),

                        ('%.5f', 'train_acc'),
                        ('%.5f', 'val_acc'),
                        )
    if emb_vis:
        hypernet.vis_embeddings(training_log.log_feature_emb)
        hypernet.vis_embeddings_out(training_log.log_feature_emb)
    m_saver= ModelSaver_in_merge(g_logger=g_logger, model_name= model_id, save_cp = save_cp)
    m_saver_B= ModelSaver_in_merge(g_logger=g_logger, model_name= model_id, save_cp = False)
    loss_CE = Value_averager()
    obs1 = Value_averager()
    obs2 = Value_averager()

    grads = None

    for epoch in tqdm(range(config.epochs_hn), desc = f"Training {model_id} / {config.dataset}"):
            speed_t = []
            epoch_time = time.time()
            # print(slmc.theta_s.from_theta_g.detach())
            for i, batches in enumerate(zip(*training_data)):
                if len(batches) == 1:
                    x, y = batches[0]
                else:
                    x = batches[0][0] #torch.cat((batches[0][0]), dim = 0)
                    y = batches[0][1] #torch.cat((batches[0][1]), dim = 0)
                    
                    x2 = batches[1][0]
                    y2 = batches[1][1]
                    
                input = x.to(device)######################
                y = y.to(device)
                
                input2 = x2.to(device)######################
                y2 = y2.to(device)
                
                hypernet.train()
                hypernet.implicitnet_train()
                optimizer_.zero_grad()
                per_itr_time = time.time()

                logits = hypernet.forward_implicitnet(input)
                # print(logits)
                loss = F.cross_entropy(logits, y.detach(), label_smoothing=0.25) / x.shape[0]
                loss.backward(retain_graph= True)
                
                hypernet_B.train()
                hypernet_B.implicitnet_train()
                
                
                logits2 = hypernet_B.forward_implicitnet(input2)
                # print(logits)
                loss2 = F.cross_entropy(logits2, y2.detach(), label_smoothing=0.25) / x.shape[0]
                loss2.backward()
                
                #  ✅ (1) Gradient norm diagnostics
                # print("🔍 Gradient norms per parameter:")
                # for name, param in hypernet.named_parameters():
                #     if param.requires_grad:
                #         if param.grad is None:
                #             print(f"{name}: ❌ No gradient")
                #         else:
                #             print(f"{name}: ✅ grad norm = {param.grad.norm().item():.4e}")
         
                optimizer_.step()

                speed_t.append(time.time() - per_itr_time)

                loss_CE.update(loss.mean().item() + loss2.mean().item()) 

                training_log.log_into_csv_(epoch+1,
                                                i,
                                                loss_CE.avg,
                                                obs1.avg, #0.,
                                                obs2.avg) #0.)
              
            # g_logger.info(f"epoch[{epoch+1}/{config.fitting_epoch}], grad (p):{grad1.avg: .6f}")
            g_logger.info(f" epoch[{epoch+1}/{config.epochs_hn}] & speed per epoch: {(time.time() - epoch_time): .5f}, Loss_CE (g -> p):{loss_CE.avg: .4f}")

            acc = prediction_vali(hypernet, testing_data[0], device)
            m_saver(acc, hypernet, config.model_save_path)
            acc2 = prediction_vali(hypernet_B, testing_data[1], device)
            m_saver_B(acc2, hypernet_B, config.model_save_path)
            
            g_logger.info("")
            g_logger.info("")
    g_logger.info("Training is done .... ")
    g_logger.info("")
    g_logger.info("Evaluation is making .... ")
    hypernet.eval()
    hypernet.implicitnet_train(False)
    hypernet = m_saver.get_best_model(hypernet,config.model_save_path)
    if emb_vis:
        hypernet.vis_embeddings_out(training_log.log_feature_emb, etc = "after")
    return hypernet, hypernet_B