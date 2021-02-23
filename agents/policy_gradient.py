import numpy as np

import time

import scipy.signal

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from agents.mlp import mlp

def discount_cumsum(x, discount):
    """
    Compute  cumulative sums of vectors.

    Input: [x0, x1, ..., xn]
    Output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, ..., xn]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PolicyNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        """
        Return a distribution over discrete actions for a given observation
        """
        return Categorical(logits=self.logits_net(obs))


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_space_dim,
                 hidden_sizes=(64,64), activation=nn.ReLU):
        super().__init__()
        self.pi = PolicyNetwork(obs_dim, act_space_dim, hidden_sizes, activation)
        self.v  = ValueNetwork(obs_dim, hidden_sizes, activation)

    def step(self, state):
        """
        Take an state and return a sampled action and value at that state
        """
        with torch.no_grad():
            action_dist = self.pi(state)
            action = action_dist.sample()
            value = self.v(state) 

        return action.item(), value.item()

    def act(self, state):
        return self.step(state)[0]


class TrajectoryBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        # Observations
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32).squeeze()
        # Actions
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32).squeeze()
        # Calculated TD residuals
        self.tdres_buf = np.zeros(size, dtype=np.float32)
        # Rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # Rewards to go
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # Values predicted
        self.val_buf = np.zeros(size, dtype=np.float32)

        # Discount factor and GAE parameter
        self.gamma = gamma
        self.lam = lam

        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the outcome observed outcome.
        """
        # Buffer has to have room so you can store
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def end_traj(self, last_val=0):
        """
        Call after a trajectory ends. Last value is value(state) if cut-off at a
        certain state, or 0 if trajectory ended uninterrupted
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # Calculate TD error for advantage estimation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.tdres_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lam)

        # Compute rewards to go
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr


    def get(self):
        """
        Call after an epoch ends. Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # Normalize TD error advantage estimates
        self.tdres_buf = self.tdres_buf - self.tdres_buf.mean()
        self.tdres_buf = self.tdres_buf / self.tdres_buf.std()

        data = dict(obs=self.obs_buf, act=self.act_buf, tdres=self.tdres_buf, ret=self.ret_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class PolicyGradientAgent:
    def __init__(self):
        self.env = None

        self.obs_dim = 8
        self.act_dim = 1
        self.act_space_dim = 4

        self.hid = 64  # layer width of networks
        self.l = 2  # layer number of networks

        # Initialize actor critic networks
        self.ac = ActorCritic(self.obs_dim, self.act_space_dim, 
                              hidden_sizes=[self.hid]*self.l)

    def set_env(self, env):
        self.env = env

    def save_model(self):
        # Hardcoded path for saving weights
        path_actor = 'weights/policy_gradient/actor_network_trained'
        path_critic = 'weights/policy_gradient/critic_network_trained'
        torch.save(self.ac.pi.state_dict(), path_actor)
        torch.save(self.ac.v.state_dict(), path_critic)
    
    def load_model(self):
        # Hardcoded path for loading weights
        self.ac.pi.load_state_dict(torch.load('weights/policy_gradient/actor_network_trained'))
        self.ac.v.load_state_dict(torch.load('weights/policy_gradient/critic_network_trained'))

    def train(self):
        """
        Training loop.
        """
        if self.env is None:
            raise('enviroment not set')

        # Training parameters
        # Number of training steps per epoch
        steps_per_epoch = 3000
        # Number of epochs to train for
        epochs = 150
        # The longest an episode can go on before cutting it off
        max_ep_len = 300

        # Discount factor
        gamma = 0.99
        # Bias variance trade-off parameter for generalized advantage estimate
        lam = 0.97

        # Optimizers
        pi_lr = 3e-3
        vf_lr = 1e-3
        pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        v_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        
        # Set up buffer
        buf = TrajectoryBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma, lam)

        # Initialize the environment
        state, ep_ret, ep_len = self.env.reset(), 0, 0

        # Collect rollouts from episodes within an epoch and update the
        # actor critic networks each epoch
        for epoch in range(epochs):
            ep_returns = []
            num_episodes = 0
            for t in range(steps_per_epoch):
                a, v = self.ac.step(torch.as_tensor(state, dtype=torch.float32))

                next_state, r, terminal = self.env.transition(a)
                ep_ret += r
                ep_len += 1

                # Store transition
                buf.store(state, a, r, v)

                # Update state (critical!)
                state = next_state

                timeout = ep_len == max_ep_len
                epoch_ended = (t == steps_per_epoch - 1)

                if terminal or timeout or epoch_ended:
                    if epoch_ended:
                        _, v = self.ac.step(torch.as_tensor(state, dtype=torch.float32))
                    else:
                        v = 0
                    if timeout or terminal:
                        ep_returns.append(ep_ret)  # only store return when episode ended
                    buf.end_traj(v)
                    state, ep_ret, ep_len = self.env.reset(), 0, 0
                    num_episodes += 1

            mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
            print(f"Epoch: {epoch+1}/{epochs}, mean return {mean_return}")


            # Get epoch trajectories
            data = buf.get()

            # Need to store the action probabilities of the old policy network
            action_probs_old = []
            for state, action in zip(data['obs'], data['act']):
                action_dist = self.ac.pi(state)
                action_idx = int(action.item())
                action_probs_old.append( action_dist.probs[action_idx].detach() )

            # Update policy network using PPO and generalized advantage estimates
            for _ in range(7):
                loss_policy = torch.Tensor([0])
                for state, action, action_prob_old, advantage in zip(data['obs'], data['act'], action_probs_old, data['tdres']):
                    action_dist = self.ac.pi(state)
                    action_idx = int(action.item())
                    r = action_dist.probs[action_idx] / action_prob_old

                    ppo_loss = torch.min(r * advantage,
                                        torch.clamp(r, min=1-0.2, max=1+0.2) * advantage )

                    loss_policy -= ppo_loss
                    loss_policy -= 0.01 * action_dist.entropy() # include an entropy regularizer term
                
                loss_policy /= num_episodes
        
                pi_optimizer.zero_grad() 
                loss_policy.backward()
                pi_optimizer.step()            

            # Update value network
            for _ in range(100):
                v = self.ac.v(data['obs'])
                v_targets = data['ret']
                loss_v = F.l1_loss(v, v_targets)

                v_optimizer.zero_grad()
                loss_v.backward()
                v_optimizer.step()

        return True


    def get_action(self, obs):
        """
        Get best action from trained policy network
        """
        with torch.no_grad():
            actions = self.ac.pi(torch.as_tensor(obs, dtype=torch.float32))
            return actions.probs.argmax().item()