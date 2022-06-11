from plot_helper import plot_perf_graphs

class Agent:
    def __init__(self, env, qtable, rewards, game_over, failure_reward):
        self.env = env
        self.qtable = qtable
        self.rewards = rewards
        self.game_over = game_over
        self.failure_reward = failure_reward

    def train(self, training_episodes):
        self.training_episodes = training_episodes
        prev_observation, info = self.env.reset(return_info=True)
        for _ in range(training_episodes):
            # env.render()
            # Step - 2 : Select an action to perform
            action = self.env.action_space.sample()
            # Step - 3 : Perform the selected action & measure the reward
            observation, reward, done, info = self.env.step(action)
            if done:
                reward = self.failure_reward
            self.rewards.add(reward, True)
            self.game_over.add(done, True)
            # Step - 4 : Update Q - table
            self.qtable.update_table(observation, prev_observation, action, reward)
            # reset game when game is over.
            if done:
                observation, info = self.env.reset(return_info=True)
            # Set current observation as previous observation for next iteration.
            prev_observation = observation
    print('Successfully completed training')

    def test(self, testing_episodes):
        self.testing_episodes = testing_episodes
        prev_observation, info = self.env.reset(return_info=True)
        for _ in range(testing_episodes):
            self.env.render()
            # Step - 2 : Select an action to perform
            action = self.qtable.best_action(prev_observation)
            # Step - 3 : Perform the selected action & measure the reward
            observation, reward, done, info = self.env.step(action)
            if done:
                reward = self.failure_reward
            self.rewards.add(reward, False)
            self.game_over.add(done, False)
            # Reset game when it is over.
            if done:
                observation, info = self.env.reset(return_info=True)
            # Set current observation as previous observation for next iteration.
            prev_observation = observation
        print('Successfuly completed testing')

    def plot_summary_graphs(self):
        plot_perf_graphs(self.testing_episodes, self.rewards, self.game_over)
    
    def print_summary(self):
        print('\n################################## SUMMARY #######################################\n')
        print('Trained for : ', self.training_episodes, ' episodes')
        print('Tested for : ', self.testing_episodes, ' episodes')
        print('Games failed: ', self.game_over.totalCount['testing'], ' times')
        print('\n##################################################################################')