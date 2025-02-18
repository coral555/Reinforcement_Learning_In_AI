import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# =======================
# question 1: monte carlo for jacky
# =======================

def monte_carlo_jacky(num_episodes=1000, gamma=1.0, alpha=0.1, epsilon=0.1):
    """
    monte carlo learning algorithm for the jacky game.
    
    args:
        num_episodes: how many times to play the game to learn.
        gamma: factor to reduce the importance of future rewards (not used here).
        alpha: how much to change the q-value after learning.
        epsilon: chance to pick a random action instead of the best one.

    returns:
        q: the table that saves the best actions.
        avg_reward: average reward after testing the learned actions.
    """
    # create a table to hold the best actions. it has 22 rows (for states 0 to 21) and 2 columns (actions: stay or hit)
    q = np.zeros((22, 2))

    for episode in range(num_episodes):
        print(f"episode {episode+1}/{num_episodes}")
        # start the game with a sum of 0
        s = 0
        trajectory = []  # saves the steps we take in the game

        while True:
            # pick an action (0 = stay, 1 = hit)
            if random.uniform(0, 1) < epsilon:
                a = random.choice([0, 1])  # random action
            else:
                a = np.argmax(q[s])  # best action from the table

            print(f"current state: {s}, action: {'stay' if a == 0 else 'hit'}")

            if a == 0:  # if stay
                trajectory.append((s, a, s))  # reward is the current sum
                break
            else:  # if hit
                card = random.randint(1, 10)  # draw a card
                s += card
                if s > 21:  # if the sum goes over 21
                    trajectory.append((s - card, a, 0))  # reward is 0 because we lost
                    break
                trajectory.append((s - card, a, None))  # game continues, no reward yet

        # learn from the game, starting from the end
        g = 0  # total reward (starts at 0)
        for s_t, a_t, r_t in reversed(trajectory):
            if r_t is not None:
                g = r_t  # if there is a reward, use it
            # update the table using this equation:
            # q(s, a) = q(s, a) + alpha * (g - q(s, a))
            q[s_t, a_t] += alpha * (g - q[s_t, a_t])

    # create a policy that picks the best action from the table
    def policy(s):
        return np.argmax(q[s])  # best action from q

    # test the policy by playing 100 games
    rewards = []
    for _ in range(100):
        s = 0
        while True:
            a = policy(s)
            if a == 0:  # if stay
                rewards.append(s)
                break
            s += random.randint(1, 10)  # if hit, add a card
            if s > 21:  # if bust
                rewards.append(0)
                break

    avg_reward = np.mean(rewards)  # average of all rewards
    print(f"average reward after evaluation: {avg_reward}")
    return q, avg_reward

# =======================
# question 2: deep q-learning for jacky with randomized rewards
# =======================

class QNetwork(nn.Module):
    """
    this is the brain of the agent. it is a neural network.
    it takes the state (sum and reward list) and gives a q-value for 'hit'.
    """
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(43, 128),  # input: 43 numbers (22 one-hot + 21 normalized rewards)
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # output: q-value for 'hit'
        )

    def forward(self, x):
        return self.fc(x)

class TrainJackyRandomized:
    """
    this trains an agent to play jacky with random rewards using deep q-learning.
    """
    def __init__(self, num_episodes=1000, gamma=1.0, epsilon=0.1, batch_size=256, lr=0.001):
        # set up the network, optimizer, and training parameters
        self.q_net = QNetwork()  # the network to learn q-values
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)  # to adjust weights
        self.loss_fn = nn.MSELoss()  # mean squared error to measure mistakes
        self.replay_buffer = deque(maxlen=100000)  # to store game data
        self.num_episodes = num_episodes
        self.gamma = gamma  # factor to reduce future rewards
        self.epsilon = epsilon  # chance to pick a random action
        self.batch_size = batch_size  # number of examples to learn from at once

    def policy(self, state):
        """
        pick an action using epsilon-greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return 1  # randomly choose 'hit'
        with torch.no_grad():
            q_value = self.q_net(torch.tensor(state, dtype=torch.float32)).item()
            return 1 if q_value > 0 else 0  # 'hit' if q-value is positive

    def train(self):
        """
        train the agent using game data.
        """
        for episode in range(self.num_episodes):
            if episode % 100 == 0:
                print(f"episode {episode+1}/{self.num_episodes}")
            state = 0  # start at 0
            r_vector = np.random.randint(1, 22, size=21)  # random rewards for this game

            # make a state vector: one-hot for sum and normalized rewards
            state_vector = np.concatenate([np.eye(22)[state], r_vector / 21])

            episode_buffer = []  # save steps of the game

            while True:
                action = self.policy(state_vector)  # pick an action
                if action == 0 or state >= 21:  # stay or end of game
                    reward = r_vector[min(state, 20)]  # pick reward
                    episode_buffer.append((state_vector, reward, None))  # no next state
                    break
                else:  # hit
                    card = np.random.randint(1, 11)  # draw a card
                    state += card
                    state = min(state, 21)  # max is 21

                    if state > 21:  # if bust
                        next_state_vector = np.zeros(43)  # no valid next state
                        reward = 0  # lose
                    else:
                        next_state_vector = np.concatenate([np.eye(22)[state], r_vector / 21])
                        reward = 0  # no reward yet

                    episode_buffer.append((state_vector, reward, next_state_vector))  # save step
                    state_vector = next_state_vector  # move to next state

            # add steps to replay buffer
            for state, reward, next_state in episode_buffer:
                self.replay_buffer.append((state, reward, next_state))

            # train the network if enough examples
            if len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
                rewards = torch.tensor([x[1] for x in batch], dtype=torch.float32)
                next_states = torch.tensor(np.array([x[2] if x[2] is not None else np.zeros(43) for x in batch]), dtype=torch.float32)

                # get q-values for current and next states
                q_values = self.q_net(states).squeeze()
                next_q_values = self.q_net(next_states).squeeze()
                
                # calculate target: reward + gamma * next q-value
                target = rewards + self.gamma * next_q_values

                # compute loss and update network
                loss = self.loss_fn(q_values, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, num_episodes=100):
        """
        test the agent.
        """
        total_reward = 0
        for _ in range(num_episodes):
            state = 0
            r_vector = np.random.randint(1, 22, size=21)  # random rewards
            state_vector = np.concatenate([np.eye(22)[state], r_vector / 21])

            while True:
                action = self.policy(state_vector)  # pick action
                if action == 0 or state >= 21:  # stop or end
                    total_reward += r_vector[min(state, 20)]  # add reward
                    break
                state += np.random.randint(1, 11)  # draw card
                state = min(state, 21)  # max is 21
        avg_reward = total_reward / num_episodes  # average
        print(f"average reward after evaluation: {avg_reward}")
        return avg_reward

# run monte carlo for question 1
q_table, avg_reward_q1 = monte_carlo_jacky()
print("question 1 - average reward:", avg_reward_q1)

# train and evaluate for question 2
trainer = TrainJackyRandomized()
trainer.train()
avg_reward = trainer.evaluate()
print(f"question 2 - average reward: {avg_reward}")
