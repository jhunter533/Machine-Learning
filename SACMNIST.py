import time
import gymnasium as gym
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import random
from copy import deepcopy
from gymnasium.spaces import Discrete, Box
num_classes=10
input_shape=(28,28,1)
trainData=datasets.MNIST(root ='data',train=True,transform=ToTensor(),download=True)
testData=datasets.MNIST(root='data',train=False,transform=ToTensor())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainLoad=DataLoader(trainData,batch_size=1,shuffle=True)
testLoad=DataLoader(testData,batch_size=1,shuffle=False)
classes=('0','1','2','3','4','5','6','7','8','9')
class MnistEnv(gym.Env):
    def __init__(self,images_per_epsiode=1,dataset=trainData,random=True):
        super(MnistEnv,self).__init__()
        self.action_space=Discrete(10)
        self.observation_space=Box(low=0,high=1,shape=(1,28,28),dtype=np.float32)
        self.images_per_epsiode=images_per_epsiode
        self.step_count=0
        self.dataset=dataset
        self.data_loader=DataLoader(dataset,batch_size=1,shuffle=random)
        self.data_iter=iter(self.data_loader)
        self.random=random
        self.current_image=None
        self.current_label=None
    def step(self,action):
        done=False
        reward=int(action==self.current_label.item())
        obs=self._next_obs()
        self.step_count+=1
        if self.step_count>=self.images_per_epsiode:
            done=True
        return obs,reward,done,{}
    def reset(self):
        self.step_count=0
        obs=self._next_obs()
        return obs
    def _next_obs(self):
        try:
            self.current_image,self.current_label=next(self.data_iter)
        except StopIteration:
            self.data_iter=iter(self.data_loader)
            self.current_image,self.current_label=next(self.data_iter)
        return self.current_image


class ReplayBuffer:
    #Replay Buffer
    #Functions as FIFO to store and sample data for SAC agent
    def __init__(self,memory_capacity=int(2e6),batch_size=256,num_actions=10,num_states=28*28):
        self.memory_capacity=memory_capacity
        self.num_states=num_states
        self.num_actions=num_actions
        self.batch_size=batch_size
        self.buffer_counter=0
        self.state_buffer=np.zeros((self.memory_capacity,self.num_states))
        self.action_buffer=np.zeros(self.memory_capacity)
        self.reward_buffer=np.zeros(self.memory_capacity)
        self.next_state_buffer=np.zeros((self.memory_capacity,self.num_states))
        self.done_buffer=np.zeros(self.memory_capacity)
    def record(self,observation,action,reward,next_observation,done):
        index = self.buffer_counter % self.memory_capacity
        #Allows index to be overwritten when memory is full
        self.state_buffer[index] = observation.flatten()
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_observation.flatten()
        self.done_buffer[index] = done
        self.buffer_counter += 1
    def sample(self):
        range1 = min(self.buffer_counter, self.memory_capacity)
        indices = np.random.randint(0, range1, size=self.batch_size)
        #returned indices are the size of batch and from any previous state action pair
        #Can be random because the policy relies on the fact that the optimal policy will solve previous and future states
        states = torch.tensor(self.state_buffer[indices], dtype=torch.float32).to(device)
        actions = torch.tensor(self.action_buffer[indices], dtype=torch.int64).to(device)
        rewards = torch.tensor(self.reward_buffer[indices], dtype=torch.float32).to(device)
        next_states = torch.tensor(self.next_state_buffer[indices], dtype=torch.float32).to(device)
        dones = torch.tensor(self.done_buffer[indices], dtype=torch.float32).to(device)
        return states,actions,rewards,next_states,dones

class Critic(nn.Module):
    #Critic class allows creation of crtic networks to evaluate the policy
    def __init__(self,num_states,num_actions,action_bound,learning_rate):
        super(Critic,self).__init__()
        self.num_actions=num_actions
        self.num_states=num_states
        self.action_bound=action_bound
        self.lC=learning_rate
        self.fc1=nn.Linear(num_states+num_actions,64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,1)
        # 400 is the hidden_state size, can be modified
    def forward(self,s,a):
        '''
            'a'| Input action-[(batch,num_actions)]
            's'| Input state-[(batch,num_states)]
            'x'| The output of each layer-[(batch,1)]
        '''
        #concatenate the state and action to combine their dimensions
        x=torch.cat((s,a),-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        #linear activation outputs 1 number 0 to 1 to evaluate policy
        x=self.fc3(x)
        return (x)
class Actor(nn.Module):
    #Actor class allows creation of actor networks to find policy
    def __init__(self,num_states,num_actions,learning_rate,action_bound):
        super(Actor,self).__init__()
        self.num_states=num_states
        self.num_actions=num_actions
        self.lA=learning_rate
        self.action_bound=action_bound
        # 400 is hidden_state size
        self.fc1=nn.Linear(num_states,64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,num_actions)
        #self.log_std_head=nn.Linear(400,num_actions)
        #log distrubution that others use
        
    def forward(self,state):
        '''
            'state'| Input state-[(batch,num_states)]
            'mu'| Output mean actions-[(batch,num_action)]
            'log_std_head'| Output log probability-[(batch,num_action)]
        '''
        state=torch.tensor(state,dtype=torch.float32).clone().detach().to(device)
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        #mu=self.mu_head(x)
        #log_std_head=(self.log_std_head(x))
        #log_std_head=torch.clamp(log_std_head,self.min_log_std,self.max_log_std)
        #return mu,log_std_head
        action_probs=F.softmax(self.fc3(x),dim=-1)
        return action_probs
class Agent:
    def __init__(self,env):
        self.env=env
        self.state_dimension=np.prod(self.env.observation_space.shape)
        self.action_dimension=self.env.action_space.n
        self.action_bound=1
        self.buffer=ReplayBuffer()
        self.learning_rate1=1e-4
        self.learning_rate2=1e-4
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
        '''
            'state'| Input state -[(batch,state_dim)]
            'action'| pi Output action to take as sampled from policy-[(batch,1)]
                ///must take just the most recent for env step
            'log_probs| Outputs the final log probability of policy of sampled action-[(batch,1)]'
        '''
        state=state.to(device)
        action_probs=self.actor(state)
        action_dist=torch.distributions.Categorical(action_probs)
        action=action_dist.sample()
        log_prob=action_dist.log_prob(action)
        return action,log_prob
        # std=torch.exp(log_std)
        #normal=torch.distributions.Normal(mu,std)
        #uAction=normal.rsample()
        #action=self.action_bound*torch.tanh(uAction)
        #log_probs=normal.log_prob(uAction).sum(axis=-1,keepdim=True)
        #transform=2*(np.log(2)-uAction-self.sp(-2*uAction)).sum(axis=-1,keepdim=True)
        #log_probs-=transform
        #return action,log_probs
    def det_action(self,state):
        '''
            'state'| Input state-[(1,num_states)]
            'action'| Output singular action
        '''
        #This function finds the deterministic action for testing in which we do not use gaussian policies but directly constrain the mean action
        state=state.unsqueeze(0).to(device)
        action_probs=self.actor(state)
        action=torch.argmax(action_probs,dim=-1)
        return action.cpu().item()
    def soft_update(self):
        #soft update function allows us to not have a value network
        #updates the target critic parameters to slowly approach the normal critic parameters
        for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(),self.critic2.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
    def train(self,max_step,max_episode):
        total_steps=0
        i=0
        for episode in range(max_episode):
            state = self.env.reset().to(device).squeeze().view(-1)
            print("/////////////////////")
            print("episode",episode)
            for step in range(max_step):
                total_steps+=1
                #if the training has just started increase exploration by randomly selecting actions
                action,_=self.action_probs(state)
                #Allows you to repeat the action a different amount of times to increase stability
                next_state,reward,done,_=self.env.step(action)
                next_state = next_state.to(device).squeeze().view(-1)
                self.buffer.record(state.cpu().numpy(),action,reward,next_state.cpu().numpy(),done)
                #Record current state action pair
                states,actions,rewards,next_states,dones=self.buffer.sample()
                #get sample batch from buffer
                actions=actions.unsqueeze(1)
                rewards=torch.unsqueeze(rewards,1)
                dones=torch.unsqueeze(dones,1)
                actions_onehot=F.one_hot(actions.long(),num_classes=self.action_dimension).squeeze(1).float().to(device)
                #unsqueeze to match dimensions [batch,1] instead of [batch,]
                q1=self.critic(states,actions_onehot)
                q2=self.critic2(states,actions_onehot)
                #get current critics
                with torch.no_grad():
                    #Calculate the value target without updating gradients
                    next_action,next_log_probs=self.action_probs(next_states)
                    #get action from policy with sample next states
                    next_action=next_action.unsqueeze(1)
                    next_action_onehot=F.one_hot(next_action.long(),num_classes=self.action_dimension).squeeze(1).float().to(device)
                    q1_next_target=self.target_critic(next_states,next_action_onehot)
                    q2_next_target=self.target_critic2(next_states,next_action_onehot)
                    q_next_target=torch.min(q1_next_target,q2_next_target)
                    #double q clip trick
                    value_target=rewards+(1-dones)*self.gamma*(q_next_target-self.alpha*next_log_probs)
                q1_loss=((q1-value_target)**2).mean()
                q2_loss=((q2-value_target)**2).mean()
                #calculate the mse loss of the critics and update their gradients
                loss_q=q2_loss+q1_loss 
                self.critic_optimizer.zero_grad()
                self.critic2_optimizer.zero_grad()
                q1_loss.backward()
                q2_loss.backward()
                self.critic_optimizer.step()
                self.critic2_optimizer.step()
                self.actor_optimizer.zero_grad()
                #Calculate the Actor loss by resampling the action based off sampled states and calculating the critic
                actions_pred,log_pred=self.action_probs(states)
                actions_pred_onehot=F.one_hot(actions_pred.long(),num_classes=self.action_dimension).squeeze(1).float().to(device)
                q1_pred=self.critic(states,actions_pred_onehot)
                q2_pred=self.critic2(states,actions_pred_onehot)
                q_pred=torch.min(q1_pred,q2_pred)
                actor_loss=(self.alpha*log_pred-q_pred).mean()
                #mse loss for actor and update gradient
                actor_loss.backward()
                self.actor_optimizer.step()
                self.soft_update()
                if done:
                    break
                state=next_state
    def evaluate(self,num_episodes=100):
        total_correct=00
        total_images=00
        for episode in range(num_episodes):
            state=self.env.reset().to(device).squeeze().view(-1)
            done=False
            while not done:
                action=self.det_action(state)
                next_state,reward,done,_=env.step(action)
                total_correct+=reward
                total_images+=1
                state=next_state
        accuracy=total_correct/total_images
        print('accuracy',accuracy)
env=MnistEnv()
agent=Agent(env)
agent.train(max_step=1,max_episode=1000000)
agent.evaluate(num_episodes=len(testData))
