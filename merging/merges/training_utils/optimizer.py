import torch
import logging
import os
import math
from torch.optim.optimizer import Optimizer

def opt_constructor(scheduler,
        models,
        lr,

        warm_up = None,
        fianl_step = None,
        start_lr = None,
        ref_lr = None,
        final_lr = None,
        start_wd = None,
        final_wd = None,
        optimizer = "adam",
        full_KD = None,
        wd_adam = None):
    log_loc = os.environ.get("log_loc")
    root_dir = os.getcwd() 
    logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all'), level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('From Optimizer')

    if not scheduler:
        lr = lr
        param_groups = []
        for model in models:
            model_param_groups = [
                {
                    'params': (p for n, p in model.named_parameters() if ('bias' not in n)),
                    'lr': lr
                },
                {
                    'params': (p for n, p in model.named_parameters() if ('bias' in n)),
                    'lr': lr
                }
            ]
    
            param_groups.extend(model_param_groups)
        
        if full_KD is not None:
            param_groups.extend([
                {
                    'params': (p for n, p in full_KD.named_parameters() if ('bias' not in n)),
                    'lr': lr * 0.01
                },
                {
                    'params': (p for n, p in full_KD.named_parameters() if ('bias' in n)),
                    'lr': lr * 0.01
                }
            ])

        opt = {'adam': torch.optim.Adam(param_groups, weight_decay= wd_adam if wd_adam is not None else 0.),
               'sgd': torch.optim.SGD(param_groups, momentum=0.90, weight_decay=wd_adam if wd_adam is not None else 0.),
               'lion': Lion(param_groups)
               }[optimizer]
        logger.info(f"Optimizer ({optimizer}) construction was successful")
        return opt, None, None

    else:
        param_groups = []
        for model in models:
            model_param_groups = [
                {
                    'params': (p for n, p in model.named_parameters() if ('bias' not in n)),
                    'WD_exclude': True if (start_wd == 0) and (final_wd == 0) else False,
                    'weight_decay': 0
                },
                {
                    'params': (p for n, p in model.named_parameters() if ('bias' in n)),
                    'WD_exclude': True,
                    'weight_decay': 0
                }
            ]
            param_groups.extend(model_param_groups)

        if full_KD is not None:
            param_groups.extend([
                {
                    'params': (p for n, p in full_KD.named_parameters() if ('bias' not in n)),
                    'lr': ref_lr * 0.05,
                    'WD_exclude': True,
                    'weight_decay': 0
                },
                {
                    'params': (p for n, p in full_KD.named_parameters() if ('bias' in n)),
                    'lr': ref_lr * 0.05,
                    'WD_exclude': True,
                    'weight_decay': 0
                }
            ])

        opt = {'adam': torch.optim.AdamW(param_groups, weight_decay= 0.0 if wd_adam else 0.),
               'sgd': torch.optim.SGD(param_groups, momentum=0.90),
               'lion': Lion(param_groups)
               }[optimizer]

        scheduler = WarmupCosineSchedule(
            opt,
            warmup_steps=warm_up,
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=fianl_step)

        wd_scheduler = None
        if (start_wd == 0) and (final_wd == 0):
            wd_scheduler = None
        else:
            wd_scheduler = CosineWDSchedule(
                opt,
                ref_wd=start_wd,
                final_wd=final_wd,
                T_max=fianl_step)
            
        logger.info("Optimizer (AdamW) with wd and lr scheduler construction was successful")
        return opt, scheduler, wd_scheduler

#############################################

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- consine annealing after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd
    

class Lion(Optimizer):
    r"""Implements Lion algorithm.
    https://github.com/gregorbachmann/scaling_mlps/blob/main/utils/optimizer.py
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
