'''
RNN based network

Source of the used code: https://github.com/timeseriesAI

A simple Recurrent (rnn, lstm, and gru) network 

'''
import torch
from torch import nn
from gradientNNs.TSNets import util_blocks
import torch.nn.functional as F

class _RNN_Base(nn.Module):
    def __init__(self, c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0, 
                 bidirectional=False, fc_dropout=0., init_weights=True, task = "", revin = True):
        super(_RNN_Base, self).__init__()
        self.task = task
        self.revin = revin
        self.rnn = nn.LSTM(c_in, hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        if task == "fault_det_rec":
            self.fc = nn.Linear(hidden_size * (1 + bidirectional), c_out)
        else:
            self.fc = nn.Linear(hidden_size * (1 + bidirectional), c_out)
        if init_weights: self.apply(self._weights_init)

    def forward(self, x): 
        # By default, we expect the input x is in shape of (batch, sequence, channels)
        if self.revin:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean
            x_std=torch.sqrt(torch.var(x, dim=1, keepdim=True)+ 1e-5)
            x = x / x_std
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        if self.task == "fault_det_rec":
            output = output
        else:
            output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.fc(self.dropout(output))
        if self.revin:
            y = y * x_std
            y = y + x_mean
        return output
    
    def _weights_init(self, m): 
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)

class LSTM_network(util_blocks.baseline_network):
    '''
    Wrapper class
    '''
    def __init__(self, in_channels, num_class, task, revin, channel_dep):
        super(LSTM_network, self).__init__(in_channels, num_class, task, revin, channel_dep)    
        self.C_true = num_class
        self.C_ = self.C_true if channel_dep else 1
        self.network = _RNN_Base(c_in = in_channels, c_out = self.C_, 
                        hidden_size=128, n_layers=3, bias=True, rnn_dropout=0.15, 
                        bidirectional=False, fc_dropout=0., init_weights=True, task = task)
