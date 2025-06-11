import time
import torch
from tqdm import tqdm
from merging.merges.training_utils.logger_ import Value_averager, Logger, ModelSaver_in_merge
from merging.merges.training_utils.optimizer import opt_constructor
from merging.merges.trainings.validation import vali
import torch.nn.functional as F
from merging.merges.training_utils.grokfast import gradfilter_ma, gradfilter_ema


def basic_hn(hypernet,
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
    g_logger.info(f"************************************ Basic HN training session ({model_id})************************************")
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
                        ('%d', 'itr'),
                        ('%.5f', 'loss_Rec')
                        )
    if emb_vis:
        hypernet.vis_embeddings(training_log.log_feature_emb)
        hypernet.vis_embeddings_out(training_log.log_feature_emb)
    # m_saver= ModelSaver_in_merge(g_logger=g_logger, model_name= model_id, save_cp = save_cp)
    loss_rec = Value_averager()
    obs1 = Value_averager()
    obs2 = Value_averager()
    
#     tracked_param_name, tracked_param = next(
#     ((name, p) for name, p in hypernet.named_parameters() if p.requires_grad), 
#     (None, None)
# )

#     if tracked_param is not None:
#         tracked_param.retain_grad()
#         g_logger.info(f"🎯 Tracking parameter: {tracked_param_name}")
    hypernet.train()
    for epoch in tqdm(range(config.epochs_hn), desc = f"Training {model_id}"):
        speed_t = []
        per_itr_time = time.time()
        
        optimizer_.zero_grad()
        loss = hypernet.forward_Hypernet("learning")
        loss.backward() 
        
        #  # ✅ (1) Gradient norm diagnostics
        # print("🔍 Gradient norms per parameter:")
        # for name, param in hypernet.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is None:
        #             print(f"{name}: ❌ No gradient")
        #         else:
        #             print(f"{name}: ✅ grad norm = {param.grad.norm().item():.4e}")
                    
        optimizer_.step()
        
        speed_t.append(time.time() - per_itr_time)
        loss_rec.update(loss.mean().item()) 
        
        training_log.log_into_csv_(epoch+1,
                                    loss_rec.avg)
        
        # g_logger.info(f"epoch[{epoch+1}/{config.fitting_epoch}], grad (p):{grad1.avg: .6f}")
        if (epoch % 20) == 0:
            g_logger.info(f" Itr [{epoch+1}/{config.epochs_hn}]: loss_rec (g -> p):{loss_rec.avg: .6f}")
            g_logger.info("")
    g_logger.info("Training is done .... ")
    g_logger.info("")
    g_logger.info("Evaluation is making .... ")
    
    hypernet.eval()
    if emb_vis:
        hypernet.vis_embeddings_out(training_log.log_feature_emb, etc = "after")
    return hypernet

def sim_basic_hn(hypernet, hypernet_B,
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
    g_logger.info(f"************************************ Basic HN training session ({model_id})************************************")
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
                        ('%d', 'itr'),
                        ('%.5f', 'loss_Rec')
                        )
    if emb_vis:
        hypernet.vis_embeddings(training_log.log_feature_emb)
        hypernet.vis_embeddings_out(training_log.log_feature_emb)
    # m_saver= ModelSaver_in_merge(g_logger=g_logger, model_name= model_id, save_cp = save_cp)
    loss_rec = Value_averager()
    obs1 = Value_averager()
    obs2 = Value_averager()
    
    hypernet.train()
    hypernet_B.train()
    for epoch in tqdm(range(config.epochs_hn), desc = f"Training {model_id}"):
        speed_t = []
        per_itr_time = time.time()
        
        optimizer_.zero_grad()
        
        # loss = hypernet.forward_Hypernet("learning")
        # loss.backward(retain_graph = True) 
        
        # loss2 = hypernet_B.forward_Hypernet("learning")
        # loss2.backward() 

        loss = hypernet.forward_Hypernet("learning")
        loss2 = hypernet_B.forward_Hypernet("learning")
        loss = loss + loss2
        loss.backward() 

        optimizer_.step()
        
        speed_t.append(time.time() - per_itr_time)
        # loss_rec.update(loss.mean().item() + loss2.mean().item()) 
        loss_rec.update(loss.mean().item() ) 
        training_log.log_into_csv_(epoch+1,
                                    loss_rec.avg)
        
        # g_logger.info(f"epoch[{epoch+1}/{config.fitting_epoch}], grad (p):{grad1.avg: .6f}")
        if (epoch % 20) == 0:
            g_logger.info(f" Itr [{epoch+1}/{config.epochs_hn}]: loss_rec (g -> p):{loss_rec.avg: .6f}")
            g_logger.info("")
    g_logger.info("Training is done .... ")
    g_logger.info("")
    g_logger.info("Evaluation is making .... ")
    
    hypernet.eval()
    if emb_vis:
        hypernet.vis_embeddings_out(training_log.log_feature_emb, etc = "after")
    return hypernet, hypernet_B

def sim_reg_hn(hypernet, hypernet_B,
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
    g_logger.info(f"************************************ Basic HN training session ({model_id})************************************")
    g_logger.info("")

    if not intrinsic_training:
        m = [hypernet, hypernet_B]
    else:
        m = [hypernet.hypernet.all_emb, hypernet_B.hypernet.all_emb]

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
                        ('%d', 'itr'),
                        ('%.5f', 'loss_Rec'),
                        ('%.5f', 'loss_grad')
                        )
    if emb_vis:
        hypernet.vis_embeddings(training_log.log_feature_emb)
        hypernet.vis_embeddings_out(training_log.log_feature_emb)
    # m_saver= ModelSaver_in_merge(g_logger=g_logger, model_name= model_id, save_cp = save_cp)
    loss_rec = Value_averager()
    loss_grad = Value_averager()
    obs1 = Value_averager()
    obs2 = Value_averager()
    
    hypernet.train()
    hypernet_B.train()
    for epoch in tqdm(range(config.epochs_hn), desc = f"Training {model_id}"):
        speed_t = []
        per_itr_time = time.time()
        
        # optimizer_.zero_grad()
        # hypernet.zero_grad()
        # loss = hypernet.forward_Hypernet("learning")
        # loss.backward(retain_graph = True) 
        # grad_A = torch.cat([p.grad.view(-1) for p in hypernet.parameters()])
        
        # hypernet_B.zero_grad()
        # loss2 = hypernet_B.forward_Hypernet("learning")
        # loss2.backward(retain_graph = True) 
        # grad_B = torch.cat([p.grad.view(-1) for p in hypernet_B.parameters()])
        # # grad_A = torch.autograd.grad(loss, hypernet.parameters(), retain_graph=True, create_graph=True)
        # # grad_B = torch.autograd.grad(loss2, hypernet_B.parameters(), retain_graph=True, create_graph=True)        
        # # grad_A = torch.cat([g.view(-1) for g in grad_A])
        # # grad_B = torch.cat([g.view(-1) for g in grad_B])
        
        # cos_sim = F.cosine_similarity(grad_A, grad_B, dim=0)
        # grad_align_loss = 1 - cos_sim  # maximize cosine similarity
        # total_loss =  loss + loss2 +grad_align_loss # loss + loss2 +
        # total_loss.backward()
        # optimizer_.step()
        
        optimizer_.zero_grad()
        loss = hypernet.forward_Hypernet("learning")
        loss2 = hypernet_B.forward_Hypernet("learning")
                
        loss.backward()
        loss2.backward()

        grad_A = torch.cat([p.grad.clone().view(-1) for p in hypernet.parameters()])
        grad_B = torch.cat([p.grad.clone().view(-1) for p in hypernet_B.parameters()])
        new_grad = pcgrad_projection(grad_A, grad_B)
        
        hypernet.zero_grad()
        # Assign projected gradient to model_a
        idx = 0
        for p in hypernet.parameters():
            if p.requires_grad:
                n = p.numel()
                p.grad = new_grad[idx:idx + n].view_as(p).clone()
                idx += n

        # total_loss =  loss + loss2 # loss + loss2 +
        # total_loss.backward()
        optimizer_.step()
        
        speed_t.append(time.time() - per_itr_time)
        # loss_rec.update(loss.mean().item() + loss2.mean().item()) 
        loss_rec.update(loss.mean().item() + loss2.mean().item()) 
        # loss_grad.update(grad_align_loss) 
        training_log.log_into_csv_(epoch+1,
                                    loss_rec.avg)
        
        # g_logger.info(f"epoch[{epoch+1}/{config.fitting_epoch}], grad (p):{grad1.avg: .6f}")
        if (epoch % 20) == 0:
            g_logger.info(f" Itr [{epoch+1}/{config.epochs_hn}]: loss_rec (g -> p):{loss_rec.avg: .6f}") # , loss_grad (g -> p):{loss_grad.avg: .6f}
            g_logger.info("")
    g_logger.info("Training is done .... ")
    g_logger.info("")
    g_logger.info("Evaluation is making .... ")
    
    hypernet.eval()
    if emb_vis:
        hypernet.vis_embeddings_out(training_log.log_feature_emb, etc = "after")
    return hypernet, hypernet_B

def pcgrad_projection(g1, g2):
    dot = torch.dot(g1, g2)
    if dot < 0:
        g1_proj = g1 - dot / (g2.norm() ** 2 + 1e-10) * g2
        return g1_proj
    return g1
