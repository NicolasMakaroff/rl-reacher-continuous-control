import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    """ Actor (Policy) Model"""
    
    def __init__(self,state_size,action_size,seed,fc1_units=128,fc2_units=128):
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)
        self.bnorm = nn.BatchNorm1d(action_size)
        self.tanh = nn.Tanh()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3,3e-3)
        self.fc2.weight.data.uniform_(-3e-3,3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self,state):
        y = F.relu(self.fc1(state))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        #y = self.bnorm(y)
        y = self.tanh(y)
        
        return y
    
    
class Critic(nn.Module):
    """ Critic (Value) Model."""
    
    def __init__(self,state_size,action_size,seed,fc1_units=128, fc2_units=128):
        """
        Initialize parameters and build model
        """
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)
        self.reset_parameters()
        self.bnorm = nn.BatchNorm1d(fc1_units)
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3,3e-3)
        self.fc2.weight.data.uniform_(-3e-3,3e-3)
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state,action):
        x  = F.relu(self.bnorm(self.fc1(state)))
        x = torch.cat((x,action),dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        # return the Q-value
        return x
        
        
        
class CriticD4PG(nn.Module):
    """ Critic D4PG model 
    https://arxiv.org/pdf/1804.08617.pdf
    """
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64,fc3_units=64, natoms=51,vmin=1,vmax=1):
        super(CriticD4PG,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size,fc2_units)
        self.fc3 = nn.Linear(fc2_units,natoms)
        delta = (vmax - vmin)/(natoms-1)
        self.register_buffer('supports',torch.arange(vmin,vmax+delta,delta))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3,3e-3)
        self.fc2.weight.data.uniform_(-3e-3,3e-3)
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state, action):
        y = F.relu(self.fc1(state))
        x = torch.cat((y,action),dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def distQ(self,distribution):
        w = F.softmax(distribution,dim=1)+self.supports
        r = w.sum(dim=1)
        return r.unsqueeze(dim=1)
    
    
    