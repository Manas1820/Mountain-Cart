# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:58:01 2020

@author:Manas gupta
"""

import gym
env = gym.make("MountainCar-v0")
env.reset()

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
MARGIN = 2000

#Q table is nothing but a table athat stores all the possible values for a given objectives
# some imp things to calculate before building a Q-table 
print(env.observation_space.high)
print(env.observation_space.low)

DISCRETE_OS_SIZE = [20]* len(env.observation_space.high)
# now we are making a sepration of descreate chunks or in simple terms we are deviding our data range in seperate parts
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

import numpy as np
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state) :
    discrete_state = (state -env.observation_space.low) /discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range (EPISODES) :
    discrete_state = get_discrete_state(env.reset())
    complete = False

    if episode % MARGIN == 0:
        render = True
        print(episode)
    else:
        render = False
        
    while not complete :
        action = np.argmax(q_table[discrete_state])    # This basically represents we have 3 actions avaible    
        new_state , reward , complete , _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        if render :
            env.render()
            
        if not complete :
            max_future_q = np.max(q_table[new_discrete_state])
            current_q    =q_table[discrete_state + (action, )]
            
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
    # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly     
        elif new_state[0] >= env.goal_position:
            print(f"We made it in episode : {episode}")
            q_table[discrete_state + (action,)] = 0 # 0 is the reward here
        
        discrete_state = new_discrete_state    
        

env.close()    