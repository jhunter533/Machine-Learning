import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Lambda,concatenate
from tensorflow.keras.optimizers import Adam
import time
import gymnasium as gym
import random
import matplotlib.pyplot as plt
#obs 0 1 are -1 to 1, obs 2 is angular vel -8 to 8, action is torque which is -2 to 2
class ReplayBuffer:
    def __init__(self,memory_capacity=1000,batch_size=64,num_actions=1,num_states=3):
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
        index=self.buffer_counter%self.memory_capacity
        self.state_buffer[index]=np.squeeze(observation)
        self.action_buffer[index]=action
        self.reward_buffer[index]=reward
        self.next_state_buffer[index]=np.squeeze(next_observation)
        self.done_buffer[index]=done
        self.buffer_counter+=1
    def sample(self):
        range=min(self.buffer_counter,self.memory_capacity)
        i=np.random.choice(range,self.batch_size)
        states=self.state_buffer[i]
        actions=self.action_buffer[i]
        rewards=self.reward_buffer[i]
        next_states=self.next_state_buffer[i]
        dones=self.done_buffer[i]
        return states,actions,rewards,next_states,dones

class Critic():
    def __init__(self,num_states,num_actions,learning_rate):
        self.num_actions=num_actions
        self.num_states=num_states
        self.model=self.create_model()
        self.lC=learning_rate
        self.opt=tf.keras.optimizers.Adam(learning_rate=self.lC)

    def create_model(self):
        inp=Input((self.num_states,))
        l1=Dense(64,activation='relu')(inp)
        l2=Dense(32,activation='relu')(l1)
        aIn=Input((self.num_actions,))
        l3=Dense(32,activation='relu')(aIn)
        l4=concatenate([l2,l3],axis=1)
        l5=Dense(16,activation='relu')(l4)
        out=Dense(1,activation='linear')(l5)
        return tf.keras.Model([inp,aIn],out)
class Actor():
    def __init__(self,num_states,num_actions,learning_rate,action_bound):
        self.num_states=num_states
        self.num_actions=num_actions
        self.lA=learning_rate
        self.action_bound=action_bound
        self.model=self.create_model()
        self.opt=tf.keras.optimizers.Adam(learning_rate=self.lA)

    def create_model(self):
        self.network=tf.keras.Sequential([
            Input((self.num_states,)),
            layers.Dense(256,activation='relu'),
            layers.Dense(256,activation='relu'),
            layers.Dense(self.num_actions,activation='tanh')
            ])
        return self.network
class Agent:
    def __init__(self,env):
        self.env=env
        self.state_dimension=self.env.observation_space.shape[0]
        self.action_dimension=self.env.action_space.shape[0]
        self.action_bound=self.env.action_space.high[0]
        self.buffer=ReplayBuffer()
        self.learning_rate=.001
        self.tau=.9
        self.gamma=.9
        self.actor=Actor(self.state_dimension,self.action_dimension,self.learning_rate,self.action_bound)
        self.critic=Critic(self.state_dimension,self.action_dimension,self.learning_rate)
        self.target_actor=Actor(self.state_dimension,self.action_dimension,self.learning_rate,self.action_bound)
        self.target_critic=Critic(self.state_dimension,self.action_dimension,self.learning_rate)

        actor_weights=self.actor.model.get_weights()
        critic_weights=self.critic.model.get_weights()
        self.target_actor.model.set_weights(actor_weights)
        self.target_critic.model.set_weights(critic_weights)
    def target_network_update(self):
        aWeight=self.actor.model.get_weights()
        tAWeight=self.target_actor.model.get_weights()
        cWeight=self.critic.model.get_weights()
        tCWeights=self.target_critic.model.get_weights()
        for i in range(len(aWeight)):
            tAWeight[i]=self.tau*np.array(aWeight[i])+(1-self.tau)*np.array(tAWeight[i])
        for i in range(len(cWeight)):
            tCWeights[i]=self.tau*np.array(cWeight[i])+(1-self.tau)*np.array(tCWeights[i])
        self.target_critic.model.set_weights(tCWeights)
        self.target_actor.model.set_weights(tAWeight)
    def train(self,max_step,max_episode):

        for episode in range(max_episode):
            state,_=self.env.reset()
            print("state",state)
            print("episode",episode)
            for step in range(max_step):
                #self.env.render()
                #stateA=np.array(state)
                
                #state_cat=np.concatenate(stateA).reshape(1,-1)
                action=self.actor.model.predict(state.reshape(1,-1))
                action+=np.random.normal(0,.1,size=self.action_dimension)
                action=np.clip(action,-self.action_bound,self.action_bound)
                next_state,reward,done,trunc,info=self.env.step(action)
                
                self.buffer.record(state,action,reward,next_state,done)
                states,actions,rewards,next_states,dones=self.buffer.sample()
                target_actions=self.target_actor.model.predict(next_states)
                target_q_values=self.target_critic.model.predict([next_states,target_actions])
                targets=rewards+self.gamma*target_q_values*(1-dones)

                with tf.GradientTape() as tape:
                    predicted_q=self.critic.model([states,actions],training=True)
                    critic_loss=tf.reduce_mean(tf.square(targets-predicted_q))
                critic_gradients=tape.gradient(critic_loss,self.critic.model.trainable_variables)
                self.critic.opt.apply_gradients(zip(critic_gradients,self.critic.model.trainable_variables))
                with tf.GradientTape() as tape:
                    actions_pred=self.actor.model(states,training=True)
                    q_v=self.critic.model([states,actions_pred],training=True)
                    actor_loss=-tf.reduce_mean(q_v)
                actor_gradients=tape.gradient(actor_loss,self.actor.model.trainable_variables)
                self.actor.opt.apply_gradients(zip(actor_gradients,self.actor.model.trainable_variables))
                self.target_network_update()
                if done:
                    break
                state=next_state
                print(reward)
                
    def test(self,max_step):
        state,_=self.env.reset()
        total_reward=00
        for step in range(max_step):
            action=self.actor.model.predict(state.reshape(1,-1))
            next_state,reward,done,_,_=self.env.step(action)
            state=next_state
            total_reward+=reward
            if done:
                break
            return total_reward
env=gym.make('Pendulum-v1',render_mode='rgb_array')
max_episode=100
max_step=200
agent=Agent(env)
agent.train(max_step,max_episode)
reward=agent.test(max_step)
print(f'Reward from test:{reward}')
env.close()
