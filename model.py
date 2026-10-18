"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    q_table = np.zeros((num_states, num_actions), dtype = np.float64)
    return q_table

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    return max(q_table[state])

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    maximum = [-1,0]
    for i in range(len(q_table[state])):
        if q_table[state][i] > maximum[0]:
            maximum[0] = q_table[state][i]
            maximum[1] = i
    return maximum[1]

# Step 4 - sample_random_action
def sample_random_action(action_space):
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    return epsilon>rng.random()

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    if should_explore(epsilon,rng):
        return sample_random_action(action_space)
    
    q_values = q_table[state]
    best_actions = np.flatnonzero(q_values == np.max(q_values))
    return int(rng.choice(best_actions))

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon * decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    if done:
        return reward
    
    return reward + gamma * max_q_value(q_table,next_state)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    return target - q_table[state][action]

# Step 10 - q_learning_update (not yet solved)
# TODO: implement

# Step 11 - interaction_step (not yet solved)
# TODO: implement

# Step 12 - run_training_episode (not yet solved)
# TODO: implement

# Step 13 - train_q_learning (not yet solved)
# TODO: implement

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

