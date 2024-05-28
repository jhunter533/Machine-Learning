import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
import pdb
import time
from torch.distributions import Normal
from copy import deepcopy
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def elapsed_time(start_time):
    return time.time()-start_time
class ReplayBuffer:
   #batch_size is 256 for more complex task
    def __init__(self,memory_capacity=2000000,batch_size=256,num_actions=4,num_states=24):
        self.memory_capacity=memory_capacity
        self.num_states=num_states
        self.num_actions=num_actions
        self.batch_size=batch_size
        self.seq_length=5
        self.buffer_counter=0
        self.state_buffer=np.zeros((self.memory_capacity,self.seq_length,self.num_states))
        self.action_buffer=np.zeros((self.memory_capacity,self.seq_length,self.num_actions))
        self.reward_buffer=np.zeros((self.memory_capacity,self.seq_length))
        self.next_state_buffer=np.zeros((self.memory_capacity,self.seq_length,self.num_states))
        self.done_buffer=np.zeros((self.memory_capacity,self.seq_length))
    def record(self,observation,action,reward,next_observation,done):
        index = self.buffer_counter % self.memory_capacity
        self.state_buffer[index] = observation
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_observation
        self.done_buffer[index] = done
        self.buffer_counter += 1
        #print('observation size',observation.shape)
        #print('action size',action.shape)
        #print('reward size',reward.shape)
        #print('nextObs size',next_observation.shape)
        #print('done size',done)
    def sample(self):
        range1 = min(self.buffer_counter, self.memory_capacity)
        indices = np.random.randint(0, range1, size=self.batch_size)
        states = torch.tensor(self.state_buffer[indices], dtype=torch.float32).to(device)
        actions = torch.tensor(self.action_buffer[indices], dtype=torch.float32).to(device)
        rewards = torch.tensor(self.reward_buffer[indices], dtype=torch.float32).to(device)
        next_states = torch.tensor(self.next_state_buffer[indices], dtype=torch.float32).to(device)
        dones = torch.tensor(self.done_buffer[indices], dtype=torch.float32).to(device)
        #print('states size',states.shape)
        #print('actions size',actions.shape)
        #print('rewards size',rewards.shape)
        #print('nextStates size',next_states.shape)
        #print('dones size',dones.shape)
        return states,actions,rewards,next_states,dones
    def reset(self):
        self.state_buffer=np.zeros((self.memory_capacity,self.seq_length,self.num_states))
        self.action_buffer=np.zeros((self.memory_capacity,self.seq_length,self.num_actions))
        self.reward_buffer=np.zeros((self.memory_capacity,self.seq_length))
        self.next_state_buffer=np.zeros((self.memory_capacity,self.seq_length,self.num_states))
        self.done_buffer=np.zeros((self.memory_capacity,self.seq_length))
        self.buffer_counter=0

class Critic(nn.Module):
    def __init__(self,num_states,num_actions,action_bound,learning_rate):
        super(Critic,self).__init__()
        self.num_actions=num_actions
        self.num_states=num_states
        self.action_bound=action_bound
        self.lC=learning_rate
        self.fc1=nn.Linear(num_states+num_actions,400)
        self.lstm=nn.LSTM(400,400,batch_first=True)
        self.fc2=nn.Linear(400,400)
        self.fc3=nn.Linear(400,1)
    def forward(self,s,a):
        #print('critic state shape',s.shape)
        #print('critic action shape',a.shape)
        x=torch.cat((s,a),-1)
        #print('critic cat shape',x.shape)
        x=F.relu(self.fc1(x))
        #print('critic 1st fc shape',x.shape)
        x,_=self.lstm(x)
        #print('critic lstm shape',x.shape)
        x=F.relu(self.fc2(x))
        #print('critic 2nd fc shape',x.shape)
        x=self.fc3(x)
        #print('critic output shape',x.shape)
        return (x)

class Actor(nn.Module):
    def __init__(self,num_states,num_actions,learning_rate,action_bound):
        super(Actor,self).__init__()
        self.num_states=num_states
        self.num_actions=num_actions
        self.lA=learning_rate
        self.action_bound=action_bound
        self.fc1=nn.Linear(num_states,400)
        self.lstm=nn.LSTM(400,400,batch_first=True)
        self.fc2=nn.Linear(400,400)
        self.mu_head=nn.Linear(400,num_actions)
        self.log_std_head=nn.Linear(400,num_actions)
        self.min_log_std=-20
        self.max_log_std=2
    def forward(self,state):
        #print('actor state shape',state.shape)
        state=torch.tensor(state,dtype=torch.float32).clone().detach().to(device)
        #print('actor state tens shape',state.shape)
        x=F.relu(self.fc1(state))
        #print('actor 1st fc shape',x.shape)
        x,_=self.lstm(x)
        #print('actor lstm shape',x.shape)
        x=F.relu(self.fc2(x))
        #print('actor 2nd fc shape',x.shape)
        mu=self.mu_head(x)
        #print('actor mu shape',mu.shape)
        log_std_head=(self.log_std_head(x))
        #print('actor log shape',log_std_head.shape)
        log_std_head=torch.clamp(log_std_head,self.min_log_std,self.max_log_std)
        return mu,log_std_head

class Agent:
    def __init__(self,env):
        self.env=env
        self.state_dimension=self.env.observation_space.shape[0]
        self.action_dimension=self.env.action_space.shape[0]
        self.action_bound=(self.env.action_space.high[0])
        self.buffer=ReplayBuffer()
        self.learning_rate1=5e-4
        self.learning_rate2=5e-4
        self.tau=.005
        self.gamma=.99
        self.alpha=.2 
        self.actor=Actor(self.state_dimension,self.action_dimension,self.learning_rate1,self.action_bound).to(device)
        self.critic=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2).to(device)
        self.target_critic=deepcopy(self.critic).to(device)
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=self.learning_rate1)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=self.learning_rate2)
        self.critic2=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2).to(device)
        self.target_critic2=deepcopy(self.critic2).to(device)
        self.critic2_optimizer=optim.Adam(self.critic2.parameters(),lr=self.learning_rate2)
        self.sp=nn.Softplus()
    def action_probs(self,state):
        state=torch.tensor(state,dtype=torch.float32)
        #print('state shape action_probs',state.shape)
        mu,log_std=self.actor(state)
        std=torch.exp(log_std)
        normal=torch.distributions.Normal(mu,std)
        uAction=normal.rsample()
        action=self.action_bound*torch.tanh(uAction)
        log_probs=normal.log_prob(uAction).sum(axis=-1,keepdim=True)
        transform=2*(np.log(2)-uAction-self.sp(-2*uAction)).sum(axis=-1,keepdim=True)
        log_probs-=transform
        #print('action shape action_probs',action.shape)
        #print('log shape action_probs',log_probs.shape)
        return action,log_probs
    def det_action(self,state):
        state=torch.tensor(state,dtype=torch.float32).to(device)
        mu,_=self.actor(state)
        action=self.action_bound*torch.tanh(mu)
        return action[:,-1,:].cpu().detach().numpy()[0]
    def soft_update(self):
        for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(),self.critic2.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
    def train(self,max_step,max_episode):
        theta_values=[]
        time_values=[]
        self.buffer.reset()
        for episode in range(max_episode):
            state,_=self.env.reset()
            print("/////////////////////")
            print("episode",episode)
            state=np.tile(state,(5,1)).reshape(1,5,-1)
            for step in range(max_step):
                print('*******new step***********')
                print('state init shape',state.shape)
                action,_=self.action_probs(state)
                action=action[:,-1,:].cpu().detach().numpy()[0]
                #print('action before step shape',action.shape)
                print('action',action)
                next_state,reward,done,trunc,info=self.env.step(action)
                #print('next_state shape',next_state.shape)
                reward=reward*5
                self.buffer.record(state,action,reward,next_state,done)
                states,actions,rewards,next_states,dones=self.buffer.sample()
                rewards=torch.unsqueeze(rewards,2)
                dones=torch.unsqueeze(dones,2)
                states=states.to(device)
                actions=actions.to(device)
                rewards=rewards.to(device)
                next_states=next_states.to(device)
                dones=dones.to(device)
                q1=self.critic(states,actions)
                q2=self.critic2(states,actions)
                #print('q1 shape',q1.shape)
                with torch.no_grad():
                    next_action,next_log_probs=self.action_probs(next_states)
                    q1_next_target=self.target_critic(next_states,next_action)
                    #print('q1_next_target shape',q1_next_target.shape)
                    q2_next_target=self.target_critic2(next_states,next_action)
                    q_next_target=torch.min(q1_next_target,q2_next_target)
                    value_target=rewards+(1-dones)*self.gamma*(q_next_target-self.alpha*next_log_probs)
                    #print('value_target shape',value_target.shape)
                q1_loss=((q1-value_target)**2).mean()
                q2_loss=((q2-value_target)**2).mean()
                loss_q=q2_loss+q1_loss 
                self.critic_optimizer.zero_grad()
                self.critic2_optimizer.zero_grad()
                loss_q.backward()
                self.critic_optimizer.step()
                self.critic2_optimizer.step()
                self.actor_optimizer.zero_grad()
                actions_pred,log_pred=self.action_probs(states)
                q1_pred=self.critic(states,actions_pred)
                q2_pred=self.critic2(states,actions_pred)
                q_pred=torch.min(q1_pred,q2_pred)
                actor_loss=(self.alpha*log_pred-q_pred).mean()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.soft_update()
                next_state_seq=np.roll(torch.from_numpy(state).cpu().detach().numpy(),-1,axis=1)
                next_state_seq[0,-1,:]=next_state
                if done:
                    break
                state=next_state_seq
    def save_model(self,actor_path,critic_path,critic_path2):
        torch.save(self.actor.state_dict(),actor_path,_use_new_zipfile_serialization=True)
        torch.save(self.critic.state_dict(),critic_path,_use_new_zipfile_serialization=True)
        torch.save(self.critic2.state_dict(),critic_path2,_use_new_zipfile_serialization=True)
        print("model saved")
    def load_model(self,actor_path,critic_path,critic_path2):
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            self.critic2.load_state_dict(torch.load(critic_path2))
            print("model loaded")

env=gym.make('BipedalWalker-v3',render_mode='human')
max_episode=700
max_step=700
agent=Agent(env)
actor_path='actor_modelSACBiLSTM3.pth'
critic_path='critic_modelSAC1BiLSTM3.pth'
critic_path2='critic_modelSAC2BiLSTM3.pth'
#agent.load_model(actor_path,critic_path,critic_path2)
agent.train(max_step,max_episode)
agent.save_model(actor_path,critic_path,critic_path2)
env.close()
