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
EPISODES = 20000
MARGIN = 500

#This is another term for exploration represents the chance 
#Higher the value of epsilon more the chances of exploration 
epsilon = 0.5 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
# Amount which we want to decay by each episode
epsilon_decay_value = epsilon /(END_EPSILON_DECAYING - START_EPSILON_DECAYING) 

# For stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
# max - is the best case 
#min - it was the worst case


#Q table is nothing but a table a that stores all the possible values for a given objectives
# some imp things to calculate before building a Q-table 
print(env.observation_space.high)
print(env.observation_space.low)

DISCRETE_OS_SIZE = [40]* len(env.observation_space.high)
# now we are making a sepration of descreate chunks or in simple terms we are deviding our data range in seperate parts
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

import numpy as np
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state) :
    discrete_state = (state -env.observation_space.low) /discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range (EPISODES) :
    discrete_state = get_discrete_state(env.reset())
    episode_reward = 0
    complete = False

    if episode % MARGIN == 0:
        render = True
        print(episode)
    else:
        render = False
        
    while not complete :
        action = np.argmax(q_table[discrete_state])    # This basically represents we have 3 actions avaible    
        new_state , reward , complete , _ = env.step(action)
        episode_reward += reward
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
        
        if END_EPSILON_DECAYING >= EPISODES >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value
        
        ep_rewards.append(episode_reward)
        if not episode % 100 :
            np.save(f"qtables/{episode}-qtable.npy", q_table)
        if not episode % MARGIN:
            average_reward = sum(ep_rewards[-MARGIN :])/len(ep_rewards[-MARGIN :])            
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-MARGIN :]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-MARGIN :]))
            print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
        

env.close()    
import matplotlib.pyplot as plt
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()
