from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)


## I added the following code on top of the default code (above) as a way to run my own cross validation with random epsilon, alpha and gamma.

while best_avg_reward <= 9.7:  ## I know I could include 9.7, but I'd like results better than the minimum :D
    agent = Agent(grid_search=True)
    avg_rewards, best_avg_reward = interact(env, agent)

print("\r\nSuccess!")

'''
Finally landed on a winning combination:
----------------------------------------
Trying: epsilon=0.0005 || alpha=0.38 || gamma=0.508
Episode 14160/20000 || Best average reward 9.731
Environment solved in 14160 episodes.
Success!
'''
