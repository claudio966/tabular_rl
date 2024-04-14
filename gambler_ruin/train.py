import gymnasium as gym
from dqn import DQN
import math
from dqn import ReplayMemory
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import random
import matplotlib
from itertools import count
import torch.optim as optim


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 1
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

nextStateTable = np.array([[[0, 0.3, 0.7, 0, 0],
                                    [0, 0.2, 0.8, 0, 0]],
                                        [[0, 0, 0.9, 0, 0.1],
                                        [0, 0, 0.8, 0, 0.2]],
                                        [[0, 0.4, 0, 0.6, 0],
                                        [0, 0.5, 0, 0.5, 0]],
                                        [[0, 0, 0, 1, 0],
                                        [0, 0, 0, 1, 0]],
                                    [[0, 0, 0.9, 0, 0.1],
                                        [0, 0, 0.8, 0, 0.2]]])
rewardsTable = np.array([[[0, 10, -5, 0, 0],
                            [0, 25, -15, 0, 0]],
                            [[0, 0, -5, 0, 15],
                                [0, 0,-15, 0, 25]],
                            [[0, 15, 0, -5, 0],
                            [0, 25, 0, -15, 0]],
                            [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                            [[0, 0, -5, 0, 10],
                            [0, 0, -15, 0, 25]]])

env = gym.make("gambler", nextStateTable=nextStateTable, rewardsTable=rewardsTable)


# Get number of actions from gym action space
n_actions = nextStateTable.shape[1]
# Get the number of state observations
n_observations = nextStateTable.shape[0]

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(2, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 1000

all_rewards = list()
for i_episode in range(num_episodes):
    reward_per_episode = 0
    # Initialize the environment and get its state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    for t in range(100):
        action = select_action(state)
        observation, reward, terminated, truncated = env.step(action[0])
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        #optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        reward_per_episode += reward

        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break
    all_rewards.append(reward_per_episode/(t+1))

cumsum_reward_storage = list()
cumsum_reward = 0
episode_idx = 1
for i in all_rewards:
    cumsum_reward += i
    cumsum_reward_storage.append(i / episode_idx)
    episode_idx += 1

print('Complete')
#plot_durations(show_result=True)
plt.plot(np.cumsum(cumsum_reward_storage))
plt.title('DQN (1 hidden layer) Learning Rewards')
plt.ylabel('Average Cumulative Reward')
plt.xlabel('Episode Index')
plt.grid()
plt.savefig('plot')
