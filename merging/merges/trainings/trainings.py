import time
import torch
from tqdm import tqdm
from merges.training_utils.logger_ import Value_averager, Logger, ModelSaver_in_merge
from merges.training_utils.optimizer import opt_constructor
from merges.trainings.validation import vali
import torch.nn.functional as F
from merges.training_utils.grokfast import gradfilter_ma, gradfilter_ema



def basic(model,
            config, 
            training_data,
            testing_data,
            device,
            g_logger,
            model_id = "",
            save_cp = True
            ):
    g_logger.info("")
    g_logger.info(f"************************************ Basic training session ({model_id})************************************")
    g_logger.info("")
    
    m = [model]
    ipe = len(training_data)
    optimizer_, lr_scheduler_, wd_scheduler_ = opt_constructor(bool(config.scheduler),
                                                        m,
                                                        lr = config.lr_,

                                                        warm_up = config.epochs* ipe * config.warm_up , #int(self.epochs* ipe * self.warm_up),
                                                        fianl_step = config.epochs* ipe, #int(self.epochs* ipe),
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
    
    m_saver= ModelSaver_in_merge(g_logger=g_logger, model_name= model_id, save_cp = save_cp)
    loss_CE = Value_averager()
    obs1 = Value_averager()
    obs2 = Value_averager()

    grads = None

    for epoch in tqdm(range(config.epochs), desc = f"Training {model_id} / {config.dataset}"):
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
                model.train()
                optimizer_.zero_grad()
                per_itr_time = time.time()

                logits = model(input)
                loss = F.cross_entropy(logits, y.detach(), label_smoothing=0.45)
                loss.backward() 

                if config.grokking:
                    ### Option 1: Grokfast (has argument alpha, lamb)
                    grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
                    ### Option 2: Grokfast-MA (has argument window_size, lamb)
                    # grads = gradfilter_ma(model, grads=grads, window_size=window_size, lamb=lamb)

                # for name, param in model.set_of_invertible_transforms.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} has a gradient")
                #     else:
                #         print(f"{name} does NOT have a gradient")

                optimizer_.step()

                speed_t.append(time.time() - per_itr_time)

                loss_CE.update(loss.mean().item()) 

                training_log.log_into_csv_(epoch+1,
                                                i,
                                                loss_CE.avg,
                                                obs1.avg, #0.,
                                                obs2.avg) #0.)
              
            # g_logger.info(f"epoch[{epoch+1}/{config.fitting_epoch}], grad (p):{grad1.avg: .6f}")
            g_logger.info(f" epoch[{epoch+1}/{config.epochs}] & speed per epoch: {(time.time() - epoch_time): .5f}, Loss_CE (g -> p):{loss_CE.avg: .4f}")

            acc = vali(model, testing_data, device)

            m_saver(acc, model, config.model_save_path)
            g_logger.info("")
            g_logger.info("")
    g_logger.info("Training is done .... ")
    g_logger.info("")
    g_logger.info("Evaluation is making .... ")
    return model