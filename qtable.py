import numpy as np

class QTable:
    def __init__(self, bin_size, action_space_size, state_min_values, state_max_values, learning_rate, discount_factor):
        self.table = dict()
        self.bin_size = bin_size
        self.action_space_size = action_space_size
        self.bins = np.array([
            np.linspace(state_min_values[0], state_max_values[0], bin_size),
            np.linspace(state_min_values[1], state_max_values[1], bin_size),
            np.linspace(state_min_values[2], state_max_values[2], bin_size),
            np.linspace(state_min_values[3], state_max_values[3], bin_size),
        ])
        self.state_size = len(state_min_values)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def convert_to_discrete_state(self, state):
        discrete_state = []
        for i in range(len(state)):
            discrete_state.append(self.bins[i][np.digitize(state[i], self.bins[i]) - 1])
        return tuple(discrete_state)
    
    def get(self, state):
        state = self.convert_to_discrete_state(state)
        if state not in self.table.keys():
            self.table[state] = np.zeros(self.action_space_size)
        return self.table[state]
    
    def set(self, state, action, q_value):
        state = self.convert_to_discrete_state(state)
        self.table[state][action] = q_value

    def get_max_q(self, state):
        q_values = self.get(state)
        return np.max(q_values)

    # Q(s, a) += learning_rate * [ Reward + discount_factor * Q(s',a') - Q(s, a) ]
    def update_table(self, current_state, prev_state, action, reward):
        existing_q_value = self.get(prev_state)[action]
        temporal_diff = reward + self.discount_factor * self.get_max_q(current_state) - existing_q_value
        new_q_value = existing_q_value + self.learning_rate * temporal_diff
        self.set(prev_state, action, new_q_value)
    
    def print(self):
        for key in self.table.keys():
            print(key, ' : ', self.table[key])

    def best_action(self, state):
        q_values = self.get(state)
        return int(np.argmax(q_values))