import os
import sys
import time
import re
import pdb
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from copy import deepcopy
import gymnasium as gym
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self,memory_capacity=int(2e6),batch_size=128,num_actions=4,num_states=24):
        self.memory_capacity=memory_capacity
        self.num_states=num_states
        self.num_actions=num_actions
        self.batch_size=batch_size
        self.buffer_counter=0
        self.state_buffer=np.zeros((self.memory_capacity,self.num_states))
        self.action_buffer=np.zeros((self.memory_capacity,self.num_actions))
        self.reward_buffer=np.zeros(self.memory_capacity)
        self.next_state_buffer=np.zeros((self.memory_capacity,self.num_states))
        self.done_buffer=np.zeros(self.memory_capacity)
    def store(self,state,action,reward,next_state,done):
        index=self.buffer_counter%self.memory_capacity
        self.state_buffer[index]=state
        self.action_buffer[index]=action
        self.reward_buffer[index]=reward
        self.next_state_buffer[index]=next_state
        self.done_buffer[index]=done
        self.buffer_counter+=1
    def sample(self):
        max_range=min(self.buffer_counter,self.memory_capacity)
        indices=np.random.randint(0,max_range,size=self.batch_size)
        states=torch.tensor(self.state_buffer[indices],dtype=torch.float32).to(device)
        actions=torch.tensor(self.action_buffer[indices],dtype=torch.float32).to(device)
        rewards = torch.tensor(self.reward_buffer[indices], dtype=torch.float32).to(device)
        next_states = torch.tensor(self.next_state_buffer[indices], dtype=torch.float32).to(device)
        dones = torch.tensor(self.done_buffer[indices], dtype=torch.float32).to(device)
        return states,actions,rewards,next_states,dones

class DQN(nn.Module):
    def __init__(self,num_states,num_actions):
        super(DQN,self).__init__()
        self.num_states=num_states
        self.num_actions=num_actions
        self.layer1=nn.Linear(self.num_states,128)
        self.layer2=nn.Linear(128,128)
        self.layer3=nn.Linear(128,self.num_actions)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        return self.layer3(x)

class Agent:
    def __init__(self,env):
        self.env=env
        self.state_dim=self.env.observation_space.shape[0]
        self.action_dim=self.env.action_space.n
        self.action_bound=None
        self.buffer=ReplayBuffer(num_actions=2,num_states=4)
        self.gamma=.99
        self.tau=.005
        self.learning_rate=1e-4
        self.epsilon=.9
        self.min_epsilon=.1
        self.e_decay=.995
        self.policy=DQN(self.state_dim,self.action_dim).to(device)
        self.target_policy=deepcopy(self.policy).to(device)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=self.learning_rate)
        self.loss_function=nn.MSELoss()
    def get_action(self,state):
        if np.random.rand()<self.epsilon:
            action=self.env.action_space.sample()
        else:
            state=torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(device)
            q_values=self.policy(state)
            action=torch.argmax(q_values,dim=1).item()
        return action
    def soft_update(self):
        for target_param,param in zip(self.target_policy.parameters(),self.policy.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
    def train(self,max_episodes):
        for i in range(max_episodes):
            state,_=self.env.reset()
            done=False
            trunc=False
            print(f"////////// {i}")
            while not (done or trunc):
                action=agent.get_action(state)
                next_state,reward,done,trunc,info=self.env.step(action)
                self.buffer.store(state,action,reward,next_state,done)
                states,actions,rewards,next_states,dones=self.buffer.sample()
                rewards=torch.unsqueeze(rewards,1)
                dones=torch.unsqueeze(dones,1)
                states=states.to(device)
                actions=actions.to(device)
                rewards=rewards.to(device)
                next_states=next_states.to(device)
                dones=dones.to(device)
                with torch.no_grad():
                    next_q=self.target_policy(next_states)
                    max_q=torch.max(next_q,dim=1)[0].unsqueeze(1)
                    target_q=rewards+(1-dones)*self.gamma*max_q
                #print(actions.shape)
                actions=torch.argmax(actions,dim=1)
                q_values=self.policy(states)
                #print(actions.shape)
                curr_q=q_values.gather(1,actions.unsqueeze(1))
                loss=self.loss_function(curr_q,target_q)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.soft_update()
                state=next_state
                if self.epsilon>self.min_epsilon:
                    self.epsilon*=self.e_decay
                    self.epsilon=max(self.min_epsilon,self.epsilon)
                if done or trunc:
                    break
    def test(self):
        done =False
        trunc=False
        state,_=self.env.reset()
        self.epsilon=0
        while not (done or trunc):
            action=self.get_action(state)
            next_state,reward,done,trunc,info=self.env.step(action)
            state=next_state
            print("reward",reward)
            if done or trunc:
                break
env=gym.make('CartPole-v1',render_mode='human')
env.metadata['render_fps']=0
agent=Agent(env)
agent.train(3000)
env.metadata['render_fps']=60
agent.test()
