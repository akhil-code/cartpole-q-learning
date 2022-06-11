import matplotlib.pyplot as plt
import numpy as np

# Plotting the rewards / performance

def plot_perf_graphs(testing_episodes, rewards, game_over):
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(1, testing_episodes, testing_episodes), rewards.counts['testing'])
    plt.xlabel('episodes')
    plt.ylabel('Rewards')
    plt.title('Testing - Rewards')

    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(1, testing_episodes, testing_episodes), game_over.counts['testing'])
    plt.xlabel('episodes')
    plt.ylabel('Games failed')
    plt.title('Testing - Games failed')
    plt.show()