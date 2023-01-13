import numpy as np 
import torch
import torch.optim as optim
import logging
import os 
import sys 

def getCi(accLog):

    mean = np.mean(accLog)
    std = np.std(accLog)
    ci95 = 1.96*std/np.sqrt(len(accLog))

    return mean, ci95

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

## Optimizer
def initial_optim(decay_option, lr, weight_decay, net, optimizer) : 
    
    if optimizer == 'adamw' : 
        optimizer_adam_family = optim.AdamW
    elif optimizer == 'adam' : 
        optimizer_adam_family = optim.Adam
    if decay_option == 'all':
        #optimizer = optimizer_adam_family(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        optimizer = optimizer_adam_family(net.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)
        
    elif decay_option == 'noVQ':
        all_params = set(net.parameters())
        no_decay = set([net.vq_layer])
        
        decay = all_params - no_decay
        optimizer = optimizer_adam_family([
                    {'params': list(no_decay), 'weight_decay': 0}, 
                    {'params': list(decay), 'weight_decay' : weight_decay}], lr=lr)
        
    return optimizer


def get_motion_with_trans(motion, velocity) : 
    '''
    motion : torch.tensor, shape (batch_size, T, 72), with the global translation = 0
    velocity : torch.tensor, shape (batch_size, T, 3), contain the information of velocity = 0
    
    '''
    trans = torch.cumsum(velocity, dim=1)
    trans = trans - trans[:, :1] ## the first root is initialized at 0 (just for visualization)
    trans = trans.repeat((1, 1, 21))
    motion_with_trans = motion + trans
    return motion_with_trans
    