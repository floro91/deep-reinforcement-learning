import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

TAU = 1e-3          # for soft update of target parameters

def hidden_init(layer):
    """
    Reused from p2
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CriticNetwork(nn.Module):
    """Critic (Value) Model."""
    
    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc2_units)
        self.fc4 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.leaky_relu(self.fcs1(state))
        x = torch.cat([x, actions], dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Network:
    def __init__(self, state_size, action_size, seed):
        """
        Initializes and handles both actor and critic network (shared weghts).
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.seed = seed
        self.actor = ActorNetwork(state_size, action_size, seed)
        self.critic = CriticNetwork(state_size, action_size, seed)

    def copy(self, network):
        """
        Copy the parameters from the provided network into current actor critic networks.
        :param network: Network instance with actor and critic models
        """
        # Actor
        for target_param, local_param in zip(self.actor.parameters(), network.actor.parameters()):
            target_param.data.copy_(local_param.data)
        
        # Critic
        for target_param, local_param in zip(self.critic.parameters(), network.critic.parameters()):
            target_param.data.copy_(local_param.data)
    
    def soft_update(self, local_network, tau=TAU):
        """Soft update model parameters for current network (actor & critic).
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_network: Local network params (actor, critic) that will be used to update target network params
            tau (float): interpolation parameter 
        """
        # Actor
        for target_param, local_param in zip(self.actor.parameters(), local_network.actor.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        # Critic
        for target_param, local_param in zip(self.critic.parameters(), local_network.critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    