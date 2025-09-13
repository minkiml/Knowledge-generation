import torch
import torch.nn as nn

class tsampler(object):
    def __init__(self, max_t = 10000, 
                 warmup = False, 
                 device = "cpu",
                 grid = False):
        super(tsampler, self).__init__()
        '''
        Sample normalized t (=unnormalized_t/max_t)
        '''
        self.grid = grid
        self.device = device
        self.max_t = max_t # whole itr = epoch * num. mb
        self.warmup = warmup #"gradual"
        self.step_size = 1 / max_t # for every 1 in domain [0,1) 
        
        self.t = torch.tensor(0.) # start from 0.
        
        if grid:
            self.domain = torch.linspace(start=0,end = 1,steps=max_t).to(device)
            self.step_size = self.domain[1] - self.domain[0]
            self.idx = 0
    def sample_current(self, itr = 0):
        if not self.grid:
            if self.warmup and (itr < self.max_t * 0.05) :
                self.t = torch.tensor([1 - (itr / self.max_t)]).to(self.device)
            else:
                # random ~ U(0,T-1)
                self.t = torch.rand(1).to(self.device) * (1. - self.step_size)
        else:
            # Randomly sample an index
            self.idx = torch.randint(0, self.domain.size(0), (1,)).item()

            # Get the corresponding value from self.domain
            self.t = self.domain[self.idx].to(self.device)
            
        return self.t.clone().detach()
 
    def sample_step(self):
        # incremental by 1 at every call
        return (self.t + self.step_size).clone().detach()
    
    def sample(self, t = None):
        if t is not None:
            return torch.tensor([t]).to(self.device)
        else:
            return torch.rand(1).to(self.device)