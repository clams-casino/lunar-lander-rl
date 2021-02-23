import numpy as np

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

from agents.mlp import mlp

class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim + 1], activation)

    def forward(self, state):
        y = self.net(state)
        value, advantages = y[...,0].unsqueeze(-1), y[...,1:]
        return value + ( advantages - torch.mean(advantages, dim=-1).unsqueeze(-1) )


class ReplayBuffer:
    """
    Buffer storing state transitions of a fixed size
    The oldest transition is removed when storing a new transition if the buffer is full
    """

    def __init__(self, obs_dim, act_dim, size):
        # states
        self.state_buf = np.zeros((size, obs_dim), dtype=np.float32).squeeze()
        # actions
        self.act_buf = np.zeros((size, act_dim), dtype=np.uint8)
        # rewards
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        # next states
        self.next_state_buf = np.zeros((size, obs_dim), dtype=np.float32).squeeze()
        # flags if the transition was to a terminal state
        self.term_state_transition = np.zeros((size, 1), dtype=bool)

        self.end_ptr, self.max_size = 0, size

    def store(self, state, act, reward, next_state, is_terminal):
        if self.end_ptr < self.max_size:
            self.state_buf[self.end_ptr] = state
            self.act_buf[self.end_ptr] = act
            self.rew_buf[self.end_ptr] = reward
            self.next_state_buf[self.end_ptr] = next_state
            self.term_state_transition[self.end_ptr] = is_terminal
            self.end_ptr += 1
        else:
            self.state_buf = np.roll(self.state_buf, -1, axis=0)
            self.act_buf = np.roll(self.act_buf, -1)
            self.rew_buf = np.roll(self.rew_buf, -1)
            self.next_state_buf = np.roll(self.next_state_buf, -1, axis=0)
            self.term_state_transition = np.roll(self.term_state_transition, -1)
            self.state_buf[-1] = state
            self.act_buf[-1] = act
            self.rew_buf[-1] = reward
            self.next_state_buf[-1] = next_state
            self.term_state_transition[-1] = is_terminal

    def sample_batch(self, batch_size):
        idx = np.random.choice(self.end_ptr, batch_size, replace=False)

        data = dict(state=self.state_buf[idx], act=self.act_buf[idx], rew=self.rew_buf[idx], 
                    next_state=self.next_state_buf[idx], is_term=self.term_state_transition[idx])
        return {k: torch.as_tensor(v) for k,v in data.items()}

    def size(self):
        return self.end_ptr


class QLearningAgent:
    def __init__(self):
        self.env = None

        self.obs_dim = 8
        self.act_dim = 1
        self.act_space_dim = 4

        self.hid = 64  # layer width of networks
        self.l = 2  # layer number of networks

        # Initialize Q networks
        self.q = DuelingQNetwork(self.obs_dim, self.act_space_dim, 
                              hidden_sizes=[self.hid]*self.l,
                              activation=nn.ReLU)

        self.target_q = DuelingQNetwork(self.obs_dim, self.act_space_dim, 
                              hidden_sizes=[self.hid]*self.l,
                              activation=nn.ReLU)

        self.target_q.load_state_dict(self.q.state_dict())

    def set_env(self, env):
        self.env = env
    
    def save_model(self):
        # Hardcoded path for saving weights
        path = 'weights/q_learning/q_network_trained'
        torch.save(self.q.state_dict(), path)

    def load_model(self):
        # Hardcoded path for loading weights
        self.q.load_state_dict(torch.load('weights/q_learning/q_network_trained'))

    def train(self):
        """
        Training loop.
        """
        if self.env is None:
            raise('enviroment not set')

        # Traning parameters
        # Num of episodes and length
        max_ep_len = 300
        num_episodes = 1500

        # Replay buffer
        replay_buf_size = 100 * max_ep_len
        batch_size = 64
        replay_start_size = 10 * batch_size

        # Epsilon greedy exploration
        epsilon_init = 1.0
        epsilon_final = 0.1
        epsilon_decay = 0.996
        epsilon = epsilon_init
        
        # Target Q network update frequency
        target_q_update_steps = 500
        target_q_update_cnt = 0

        # Discount factor
        gamma = 0.99

        # Optimizer
        q_lr = 0.0005
        q_optimizer = Adam(self.q.parameters(), lr=q_lr)

        # Replay buffer
        buf = ReplayBuffer(self.obs_dim, self.act_dim, replay_buf_size)
        
        print('Training for {} episodes'.format(num_episodes))
        mean_ep_reward, ep_cnt, ep_print_freq = 0, 0, 100
        for ep in range(num_episodes):

            # Reset the environment for each episode
            state = self.env.reset()
            
            ep_reward = 0

            for step in range(max_ep_len):

                a = self.get_training_action(
                        torch.as_tensor(state, dtype=torch.float32), 
                        epsilon)

                next_state, r, terminal = self.env.transition(a)

                ep_reward += r

                # Store in replay buffer
                buf.store(state, a, r, next_state, terminal)

                # Update state (critical!)
                state = next_state


                # Sample random minibatch from replay buffer
                if buf.size() >= replay_start_size:
                    minibatch = buf.sample_batch(batch_size)

                    q_estimates_batch = torch.gather(self.q(minibatch['state']), dim=1,
                                                    index=minibatch['act'].long())

                    with torch.no_grad():
                        # Double DQN
                        # Choose greedy action using current Q network
                        a_max = torch.argmax(self.q(minibatch['next_state']), dim=1).unsqueeze(-1)

                        # Evalute Q value of greedy action using target Q network
                        next_state_value_estimates = torch.gather(self.target_q(minibatch['next_state']), dim=1,
                                                                index=a_max.long())

                        targets_batch = (minibatch['rew'] + gamma * ~minibatch['is_term'] * next_state_value_estimates).detach()


                    # Optimize Q network
                    q_optimizer.zero_grad()
                    loss = F.smooth_l1_loss(q_estimates_batch, targets_batch)
                    loss.backward()
                    q_optimizer.step()

                    # Copy over weights to target network here if enough steps have been taken
                    if target_q_update_cnt == target_q_update_steps:
                        self.target_q.load_state_dict(self.q.state_dict())
                        target_q_update_cnt = 0
                    else:
                        target_q_update_cnt += 1

                # Move to the next episode if terminated
                if terminal:
                    break
            
            # Anneal epsilon after each episode
            epsilon = max(epsilon * epsilon_decay, epsilon_final)

            mean_ep_reward += ep_reward
            ep_cnt += 1
            if ep_cnt != 0 and ep_cnt % ep_print_freq == 0:
                print('Episode {} to {} - Average return {}'.format(ep_cnt - ep_print_freq, ep_cnt, 
                                                        mean_ep_reward/ep_print_freq))
                mean_ep_reward = 0
                

    def get_training_action(self, obs, epsilon):
        """
        Choose action during training using epsilon greedy policy
        """
        if np.random.uniform() < epsilon:
            return np.random.choice(4)
        else:
            return self.get_action(obs)


    def get_action(self, obs):
        """
        Get greedy action using Q network
        """
        with torch.no_grad():
            q_values = self.q(torch.as_tensor(obs, dtype=torch.float32))
        return q_values.argmax().item()