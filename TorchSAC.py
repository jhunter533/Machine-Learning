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
class ReplayBuffer:
    def __init__(self,memory_capacity=1000000,batch_size=64,num_actions=1,num_states=3):
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
    def record(self,observation,action,reward,next_observation,done):
        index = self.buffer_counter % self.memory_capacity
        self.state_buffer[index] = observation
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_observation
        self.done_buffer[index] = done
        self.buffer_counter += 1
    def sample(self):
        range1 = min(self.buffer_counter, self.memory_capacity)
        indices = np.random.randint(0, range1, size=self.batch_size)
        states = torch.tensor(self.state_buffer[indices], dtype=torch.float32)
        actions = torch.tensor(self.action_buffer[indices], dtype=torch.float32)
        rewards = torch.tensor(self.reward_buffer[indices], dtype=torch.float32)
        next_states = torch.tensor(self.next_state_buffer[indices], dtype=torch.float32)
        dones = torch.tensor(self.done_buffer[indices], dtype=torch.float32)
        return states,actions,rewards,next_states,dones

class Critic(nn.Module):
    def __init__(self,num_states,num_actions,action_bound,learning_rate):
        super(Critic,self).__init__()
        self.num_actions=num_actions
        self.num_states=num_states
        self.action_bound=action_bound
        
        self.lC=learning_rate
        self.fc1=nn.Linear(num_states,200)
        self.fc2=nn.Linear(num_actions,200)
        self.combinedfc1=nn.Linear(400,300)
        self.combinedfc2=nn.Linear(300,1)
    def forward(self,s,a):
        state_out=F.relu(self.fc1(s))
        action_out=F.relu(self.fc2(a))
        combined=torch.cat([state_out,action_out],dim=-1)
        combined=F.relu(self.combinedfc1(combined))
        x=self.combinedfc2(combined)
        return (x)

class Actor(nn.Module):
    def __init__(self,num_states,num_actions,learning_rate,action_bound):
        super(Actor,self).__init__()
        self.num_states=num_states
        self.num_actions=num_actions
        self.lA=learning_rate
        self.action_bound=action_bound
        self.fc1=nn.Linear(num_states,400)
        self.fc2=nn.Linear(400,300)
        self.mu_head=nn.Linear(300,num_actions)
        self.log_std_head=nn.Linear(300,num_actions)
        self.min_log_std=-20
        self.max_log_std=2
    def forward(self,state):
        state=torch.tensor(state,dtype=torch.float32).clone().detach()
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        mu=self.mu_head(x)
        log_std_head=F.relu(self.log_std_head(x))
        log_std_head=torch.clamp(log_std_head,self.min_log_std,self.max_log_std)
        return mu,log_std_head

class Agent:
    def __init__(self,env):
        self.env=env
        self.state_dimension=self.env.observation_space.shape[0]
        self.action_dimension=self.env.action_space.shape[0]
        self.action_bound=(self.env.action_space.high[0])
        self.buffer=ReplayBuffer()
        self.learning_rate1=.0001
        self.learning_rate2=.001
        self.tau=.005
        self.gamma=.9
        self.alpha=.2 
        self.actor=Actor(self.state_dimension,self.action_dimension,self.learning_rate1,self.action_bound)
        self.critic=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2)
        self.target_critic=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2) 
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=self.learning_rate1)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=self.learning_rate2)
        self.critic2=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2)
        self.target_critic2=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2)
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer=optim.Adam(self.critic.parameters(),lr=self.learning_rate2)
    def select_action(self,state):
        mu,log_std=self.actor(state)
        std=torch.exp(log_std)
        action=torch.tanh(torch.normal(mu,std))
        return action
    def log_probs(self,state):
        state=torch.tensor(state,dtype=torch.float32)
        mu,log_std=self.actor(state)
        std=torch.exp(log_std)
        normal=torch.distributions.Normal(mu,std)
        action=torch.tanh(normal.sample())
        log_probs=normal.log_prob(action).sum(axis=-1,keepdim=True)
        log_probs-=torch.log(1-action.pow(2)+1e-6)
        return log_probs
    def soft_update(self):
        for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(),self.critic2.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
    def train(self,max_step,max_episode):
        theta_values=[]
        time_values=[]
        for episode in range(max_episode):
            state,_=self.env.reset()
            print("/////////////////////")
            print("episode",episode)
            for step in range(max_step):
                action=self.select_action(state).detach().numpy()
                action=np.clip(action,-self.action_bound,self.action_bound)
                print('action',action)
                #print('action',action.shape)
                next_state,reward,done,trunc,info=self.env.step(action)
                self.buffer.record(state,action,reward,next_state,done)
                states,actions,rewards,next_states,dones=self.buffer.sample()
                states=torch.FloatTensor(states)
                actions=torch.FloatTensor(actions)
                rewards=torch.FloatTensor(rewards)
                next_states=torch.FloatTensor(next_states)
                dones=torch.FloatTensor(dones)
                log_probs=self.log_probs(state)
                #print('states',states)
                #print('actions',actions)
                #print('log_probs',log_probs)
                q1=self.critic(states,actions)
                q2=self.critic2(states,actions)
                #print('q1',q1)
                with torch.no_grad():
                    next_action=self.select_action(next_states)
                    #print('next_action',next_action)
                    q1_next_target=self.target_critic(next_states,next_action)
                    q2_next_target=self.target_critic2(next_states,next_action)
                    q_next_target=torch.min(q1_next_target,q2_next_target)
                    #print('q1_next',q1_next_target)
                    next_log_probs=self.log_probs(next_states)
                    value_target=rewards+(1-dones)*self.gamma*(q_next_target-self.alpha*next_log_probs)
                    #print('value_target',value_target)
                q1_loss=((q1-value_target)**2).mean()
                q2_loss=((q2-value_target)**2).mean()
                loss_q=q2_loss+q1_loss 
                self.critic_optimizer.zero_grad()
                self.critic2_optimizer.zero_grad()
                loss_q.backward()
                self.critic_optimizer.step()
                self.critic2_optimizer.step()

                self.actor_optimizer.zero_grad()
                actions_pred=self.select_action(states)
                log_pred=self.log_probs(states)
                q1_pred=self.critic(states,actions_pred)
                q2_pred=self.critic2(states,actions_pred)
                q_pred=torch.min(q1_pred,q2_pred)
                actor_loss=(self.alpha*log_pred-q_pred).mean()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.soft_update()
                if done:
                    break
                #print(reward)
                state=next_state


    #ignore below for now
    def test(self,max_step):
        state,_=self.env.reset()
        total_reward=0
        theta_values=[]
        step_values=[]
        for step in range(max_step):
            state_tensor=torch.tensor(state,dtype=torch.float32)
            action_dist=self.actor(state_tensor)
            action=action_dist.sample().detach().numpy()
            #action=self.actor(torch.tensor(state,dtype=torch.float32)).detach().numpy()
            action =np.clip(action,-self.action_bound,self.action_bound)

            next_state,reward,done,_,_=self.env.step(action)
            state=next_state
            total_reward+=reward
            theta=np.arccos(state[0])
            theta_values.append(np.degrees(theta))
            step_values.append(step)
            if done:
                break
        plt.plot(step_values,theta_values)
        plt.xlabel('step')
        plt.ylabel('Angle (degrees)')
        plt.title('angle vs step')
        plt.grid(True)
        plt.savefig('anglevsstepTorch.png')
        plt.close()
        return total_reward
    def save_model(self,actor_path,critic_path1,critic_path2):
        torch.save(self.actor.state_dict(),actor_path,_use_new_zipfile_serialization=True)
        torch.save(self.critic1.state_dict(),critic_path1,_use_new_zipfile_serialization=True)
        torch.save(self.critic2.state_dict(),critic_path2,_use_new_zipfile_serialization=True)
        print("model saved")
    def load_model(self,actor_path,critic_path1,critic_path2):
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic1.load_state_dict(torch.load(critic_path1))
            self.critic2.load_state_dict(torch.load(critic_path2))
            print("model loaded")
            
    

env=gym.make('Pendulum-v1',render_mode='human')
max_episode=70
max_step=200
agent=Agent(env)
actor_path='actor_modelSAC.pth'
critic_path1='critic_modelSAC1.pth'
critic_path2='critic_modelSAC2.pth'
#agent.load_model(actor_path,critic_path1,critic_path2)
agent.train(max_step,max_episode)
agent.save_model(actor_path,critic_path1,critic_path2)
max_step=300
#reward=agent.test(max_step)
print(f'Reward from test:{reward}')

env.close()
