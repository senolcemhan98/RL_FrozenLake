# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:50:33 2020

@author: senol
"""
# import Libraries
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gym.envs.registration import register

#%% Create Enviroment
env = gym.make('FrozenLake-v0')


register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

#%%
# init q_table and set hyperparameters

# Q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameter
gamma = 0.90  #discount factor
alpha = 0.4   # lr
epsilon = 0.1

#%%

# Plotting Metrix
reward_list = []

episode_number = 100000
for i in range(1,episode_number+1):
    
    state = env.reset()
    
    reward_count = 0
    # for step in range(100):
    while True:
         
        # Choose an action 
        # exploit vs explore to find action
        # %10 = explore, %90 exploit        
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # Perform action
        new_state, reward, done, _ = env.step(action)
        
        # q_learning function & q_table update
        q_table[state,action] = (1-alpha) * q_table[state,action] + alpha*(reward+gamma*np.max(q_table[new_state]))
            
        # update state
        state = new_state
        
        reward_count += reward 
        
        if done:
            break
                
    if i%10 == 0:
        reward_list.append(reward_count)
        print("Episode: {}, reward {}".format(i,reward_count))
        
        

plt.plot(reward_list)
        
        
        
        
            
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        