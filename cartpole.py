from os import stat
import gym
import numpy as np
from qtable import QTable
from rewards import Reward
from game_over import GameOver
from qlearning import Agent

# constants
learning_rate = 0.1
discount_factor = 0.4
failure_reward = -250

# controls
training_episodes = 500000
testing_episodes = 5000

# making states discrete
bin_size = 30
state_min_values = [-2.5, -10, -0.418, -10]
state_max_values = [2.5, 10, 0.418, 10]

rewards = Reward()
game_over = GameOver()

# initializing open AI env.
env = gym.make('CartPole-v1')
# size of action space and observation space
action_space_size = env.action_space.n 
observation_space_size = env.observation_space.shape[0]

# Initialize Q - table
qtable = QTable(bin_size, action_space_size, state_min_values, state_max_values, learning_rate, discount_factor)
# Creating q-learning agent
agent = Agent(env, qtable, rewards, game_over, failure_reward)
# Training the agent.
agent.train(training_episodes)
# Testing the agent.
agent.test(testing_episodes)
# Print test summary
agent.print_summary()
# Plot the performance graphs
# agent.plot_summary_graphs()
# Suspend open AI env
env.close()

