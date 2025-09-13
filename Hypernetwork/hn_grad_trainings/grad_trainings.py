import time
import torch
from tqdm import tqdm
from merging.merges.training_utils.logger_ import Value_averager, Logger, ModelSaver_in_merge
from merging.merges.training_utils.optimizer import opt_constructor
from Hypernetwork.hn_trainings.validation import prediction_vali
import torch.nn.functional as F

def prediction_grad_train(hypernet,
            config, 
            training_data,
            testing_data,
            device,
            g_logger,
            model_id = "",
            save_cp = True,
            emb_vis = False,
            intrinsic_training = False,
            h = 1, # forward_step
            sampler = None,
            grad_learning = False
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
    max_ipe = config.epochs_hn* ipe
    # print(len(optimizer_.param_groups))
    # for group in optimizer_.param_groups:
    #     print("Learning rate:", group['lr'])
    #     print("Weight decay:", group.get('weight_decay', 'N/A'))
    #     print("Parameters:", group['params'])  # list of tensors
    # raise NotImplementedError("")
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
    itr = 0.
    
    def iterative_grad_train():
        # Estimate params for t + h
        t_h = sampler.sample_step()
        params_h = hypernet.forward_Hypernet(mode = "generating", t=t_h) # TODO sample from t_h ~ [0, T] ( > t_1)
        
        # compute grad manually by params_h - params_t and compute loss between gt grad and estimated grad 
        mse_loss = torch.tensor(0.).to(device)
        for n, (p_h, p_t, g_gt) in enumerate(zip(params_h, params_t, gt_grad)): 
            grad_pred = p_h - p_t
            # layer-wise mse loss
            # mse_loss += ((grad_pred - g_gt)**2).mean()
            mse_loss += (torch.abs(grad_pred - g_gt)).mean()

        mse_loss = mse_loss #/ (n+1)
        mse_loss.backward()
        optimizer_.step()
    
    def param_update(param_t, grad_t, esti_param_t, grad_learning = False):
        updated_param = []
        mae = torch.tensor(0.).to(device)
        if not grad_learning:
            for w, grad, pred in zip(param_t, grad_t, esti_param_t):
                updated_w = (w + grad).detach().clone()
                
                mae += ((pred - updated_w)**2).mean() # (torch.abs(pred - updated_w)).mean()
                
                updated_param.append(updated_w)
        else:
            for w, grad, pred in zip(param_t, grad_t, esti_param_t):                
                mae += (torch.abs(pred - grad)).mean()
                
                updated_w = (w + grad).detach().clone()
                updated_param.append(updated_w)
        return updated_param, mae
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
                hypernet.zero_grad()
                
                per_itr_time = time.time()

                t = sampler.sample_current(itr = itr)
                # TODO time 1
                logits = hypernet.forward_implicitnet(input, t = t, isolate_hypernet = True, 
                                                      grad_learning = grad_learning) # TODO sample from t_1 ~ [0, T]
                loss = F.cross_entropy(logits, y.detach(), label_smoothing=0.25) #/ x.shape[0]
                
                
                # @@@@@@@@@@@@@@@@@@@@
                # # TODO Implement h steps update of the "mainnet" given the first step weights 
                mse_loss = torch.tensor(0.).to(device)
                hypernet.zero_grad()
                optimizer_.zero_grad()
                params_t_next = None
                for i in range(h):
                    if i > 0:
                        # compute loss
                        logits = hypernet.forward_implicitnet(input, t = t_h, isolate_hypernet = True, params = params_t_next)
                        loss = F.cross_entropy(logits, y.detach(), label_smoothing=0.25) #/ x.shape[0]
                        
                    # Compute ground truth grad  # TODO time 2
                    gt_grad, params_t = hypernet.compute_gt_grad(loss, lr = config.lr_)
                    
                    # Estimate param_t+i
                    if not grad_learning:
                        t_h = sampler.sample_step()
                    else:
                        t_h = sampler.sample(t = (itr + 1) / max_ipe) # TODO t_h is incremental from 0 to 1
                                    
                    params_h = hypernet.forward_Hypernet(mode = "generating", t=t_h, grad_learning = grad_learning) # TODO sample from t_h ~ [0, T] ( > t_1)
                    
                    # Manual update (for ground-truth param_t+i) and computing loss at t+i
                    params_t_next, mse_loss = param_update(params_t, gt_grad, params_h, grad_learning = grad_learning)
                    
                    # # Estimate param_t+i
                    # t_h = sampler.sample_step()
                    # params_h = hypernet.forward_Hypernet(mode = "generating", t=t_h) # TODO sample from t_h ~ [0, T] ( > t_1)
                    
                    # mse_loss += mae
                    
                    mse_loss.backward(retain_graph = True)
                # mse_loss = mse_loss #/ (n+1)
                # mse_loss.backward()
                optimizer_.step()
                # @@@@@@@@@@@@@@@@@@@@
                
                
                
                # # Compute ground truth grad  # TODO time 2
                # gt_grad, params_t = hypernet.compute_gt_grad(loss, lr = config.lr_)
                # # TODO if we add h steps update version, we could compute the gt param_h directly (i.e., use eq 3 without computing grad)
                
                # # Estimate params for t + h
                # hypernet.zero_grad()
                # optimizer_.zero_grad()
                # t_h = sampler.sample_step()
                # params_h = hypernet.forward_Hypernet(mode = "generating", t=t_h) # TODO sample from t_h ~ [0, T] ( > t_1)
                
                # # compute grad manually by params_h - params_t and compute loss between gt grad and estimated grad 
                # mse_loss = torch.tensor(0.).to(device)
                # for n, (p_h, p_t, g_gt) in enumerate(zip(params_h, params_t, gt_grad)): 
                #     grad_pred = p_h - p_t
                #     # layer-wise mse loss
                #     # mse_loss += ((grad_pred - g_gt)**2).mean()
                #     mse_loss += (torch.abs(grad_pred - g_gt)).mean()

                # mse_loss = mse_loss #/ (n+1)
                # mse_loss.backward()
                # optimizer_.step()
                
                
                speed_t.append(time.time() - per_itr_time)
                loss_CE.update(mse_loss.mean().item()) 

                training_log.log_into_csv_(epoch+1,
                                                i,
                                                loss_CE.avg,
                                                obs1.avg, #0.,
                                                obs2.avg) #0.)
                
                itr += 1
                if grad_learning:
                    hypernet.reset_initparams(params_t_next)
            # hypernet.reset_initparams(params_h)
            
            # g_logger.info(f"epoch[{epoch+1}/{config.fitting_epoch}], grad (p):{grad1.avg: .6f}")
            g_logger.info(f" epoch[{epoch+1}/{config.epochs_hn}] & speed per epoch: {(time.time() - epoch_time): .5f}, Loss_MSE:{loss_CE.avg: .8f}")
            
            t = sampler.sample_current() # pass a desired t, otherwise it samples randomly
            acc = prediction_vali(hypernet, testing_data, device, t = t)

            m_saver(acc, hypernet, config.model_save_path, t=t)
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
