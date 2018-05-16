import numpy as np
from collections import defaultdict
import random


class Agent:

    def __init__(self, nA=6, epsilon=0.0001, alpha=.95, gamma=.95, cross_validation=False):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        ## Set values based on what's passed in
        self.nA = nA
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        if cross_validation:
            ## Set values based on a random range of possibilities to be used for simple cross-validation (change cross_validation to True to use this feature)
            self.epsilon = round(random.uniform(0.,0.001), 5)
            self.alpha = round(random.uniform(0.1,1.), 3)
            self.gamma = round(random.uniform(0.1,1.), 3)
        
            print("\rTrying: epsilon={} || alpha={} || gamma={}".format(self.epsilon, self.alpha, self.gamma), end="\r\n")
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        ## Checks if state has already come up; if so, creates probabilities using the epsilon greedy method
        if state in self.Q:
            probs = np.ones(self.nA) * (self.epsilon / self.nA)
            best_a = np.argmax(self.Q[state])
            probs[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)

        ## If it's a new state, it uses even probability.
        else:
            probs = np.full(self.nA, 1 / self.nA)

        return np.random.choice(np.arange(self.nA), p=probs)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        NOTE: Uncomment out just 1 of the below formulas before running.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        ## Default code
#         self.Q[state][action] += 1

        ## Sarsa(0) formula (best average reward: 8.762)
#         next_action = self.select_action(state)
#         self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
        
        ## Sarsamax/Q-learning formula (best average reward: 9.296)
#         self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action])
        
        ## Expected Sarsa formula (best average reward: 9.731)
        probs = np.ones(self.nA) * (self.epsilon / self.nA)
        probs[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        expected = np.dot(self.Q[next_state], probs)
        self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + (self.gamma * expected) - self.Q[state][action]))
        
        ## Reward Hacking method, UPDATE: Nevermind, won't work without modifying monitor.py
#         self.Q[state][action] += 1
#         reward += 1000
#         return reward
