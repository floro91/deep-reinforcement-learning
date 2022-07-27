from collections import deque
import numpy as np
import random
import torch
from torch import optim

from model_final import *
from buffer_final import *
from noise_final import *

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1       # step frequency for target network update
NO_UPDATES = 1         # 10 updates at a 


class Agent:

    memory = None # Shared memory for both agents
    
    def __init__(self, state_size, action_size, random_seed, action_low=-1, action_high=1):
        """
        Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            action_low (float): Minimum value for action
            action_high (float): Maxmimum value for aciton
        """

        self.seed = random.seed(random_seed)
        self.state_size = state_size
        self.action_size = action_size
        self.a_low = action_low
        self.a_high = action_high
        
        self.ACTOR_LR = LR_ACTOR
        self.CRITIC_LR = LR_CRITIC
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.TIMES_UPDATE = NO_UPDATES
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        # Local network (Actor & Critic)
        self.local_network = Network(state_size, action_size, random_seed)
        
        self.actor_optimizer = optim.Adam(self.local_network.actor.parameters(), lr=self.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.local_network.critic.parameters(), lr=self.CRITIC_LR, weight_decay=self.WEIGHT_DECAY)
        # Target network (Actor & Critic)
        self.target_network = Network(state_size, action_size, random_seed)
        
        # Noise Process
        self.ounoise = OUNoise(action_size, random_seed)
        
        # Replay memory (shared for both if no single memory per agent)
        if Agent.memory == None:
            Agent.memory = ReplayBuffer(self.BUFFER_SIZE,self.BATCH_SIZE)
        
        self.t_step = 0
    
    def act(self, state, add_noise=True):
        """Returns action for given state."""
        self.local_network.actor.eval()
        with torch.no_grad():
            action = self.local_network.actor(state)
            action = action.data.cpu().numpy()
        self.local_network.actor.train()
        if add_noise:
            return self.ounoise.get_action(action)
        return action
        
    def add_memory(states0, actions0, rewards0, next_states0, dones0, 
                   states1, actions1, rewards1, next_states1, dones1):
        """Add experience of both agents to memory."""
        Agent.memory.add(states0, actions0, rewards0, next_states0, dones0, 
                   states1, actions1, rewards1, next_states1, dones1)
    
    def step(self, agent_num):
        """Check if learning required, then use random sample from buffer to learn."""
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY

        if len(self.memory) > self.BATCH_SIZE and self.t_step == 0:
            for i in range(self.TIMES_UPDATE):
                experiences = self.memory.sample()
                self.learn(experiences, agent_num)
                self.target_network.soft_update(self.local_network)
    
    def learn(self, experiences, agent_num):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            agent_num: Agent number
        """
        states0, actions0, rewards0, next_states0, dones0, states1, actions1, rewards1, next_states1, dones1 = experiences
        if agent_num == 0:
            states = states0
            actions = actions0
            rewards = rewards0
            next_states = next_states0
            dones = dones0
        else:
            states = states1
            actions = actions1
            rewards = rewards1
            next_states = next_states1
            dones = dones1
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models        
        next_actions = self.target_network.actor(next_states)
        Q_target_next = self.target_network.critic(next_states, next_actions)
        # Compute Q targets for current states (y_i)
        Q_target = rewards + (self.GAMMA * Q_target_next * (1-dones))
        # Compute critic loss
        Q_predicted = self.local_network.critic(states, actions)
        critic_loss = F.mse_loss(Q_predicted, Q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_network.critic.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss       
        actions_pred = self.local_network.actor(states)
        actor_loss = -self.local_network.critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()