import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from algo.memory import ScalableReplayBuffer
from algo.misc import soft_update, LinearSchedule
from algo.model import DuelingDQN


class PPDQNTrainer:
    def __init__(self, graph, p,
                 num_action,
                 tau=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 batch_size=32,
                 epsilon_decay=int(1e6),
                 buffer_size=int(1e6)):
        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_action = num_action
        self.batch_size = batch_size

        self.state_dim = len(self.nodes)
        self.q_network = DuelingDQN(self.state_dim, num_action)
        self.target_network = DuelingDQN(self.state_dim, num_action)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.tau = tau
        self.gamma = gamma
        self.replay_buffer = ScalableReplayBuffer(capacity=buffer_size)
        self.schedule = LinearSchedule(epsilon_decay, epsilon_end, epsilon_start)

    def epsilon(self, t):
        return self.schedule.value(t)

    def choose_action(self, state, mask, episode=None):
        if episode is not None:
            if np.random.uniform() < self.epsilon(episode):
                num_action = len(np.where(mask != -np.inf)[0])
                return np.random.choice(num_action)

        state_tensor = torch.tensor(state, dtype=torch.int32).unsqueeze(0)
        q_values = self.q_network(state_tensor)[0].detach().numpy()
        q_values += mask
        # print(q_values, end=', ')
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def update_q_network(self, t, update_rate=100):
        if t < int(1e4) or t % 100 != 0: return None

        # Sample a batch of experiences from the buffer
        batch = self.replay_buffer.sample_(self.batch_size, num_keys=1)
        if len(batch) <= 0: return None

        states, masks, actions, rewards, next_states, next_masks, dones = zip(*batch)
        # Convert to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.int32)
        masks_tensor = torch.tensor(np.array(masks), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.int32)
        next_masks_tensor = torch.tensor(np.array(next_masks), dtype=torch.float32)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32)

        # Compute Q values using current Q network
        q_values = self.q_network(states_tensor)
        # Get the Q values for the actions taken
        action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        # Double DQN target
        next_q_values_online = self.q_network(next_states_tensor)
        next_actions = (next_q_values_online+next_masks_tensor).argmax(1, keepdim=True)
        next_q_values_target = self.target_network(next_states_tensor)
        next_q_values_selected = next_q_values_target.gather(1, next_actions).squeeze(1)
        target_q_values = rewards_tensor + self.gamma * next_q_values_selected * (1 - dones_tensor)
        # Compute loss
        loss = F.mse_loss(action_q_values, target_q_values.detach())
        # Back propagate to compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
        # Update the model parameters with the gradients
        self.optimizer.step()

        if t % update_rate == 0:
            soft_update(self.target_network, self.q_network, self.tau)
        return loss.item()

    def save_model(self, save_path):
        torch.save(self.q_network.state_dict(), save_path)

    def load_mode(self, load_path):
        state_dict = torch.load(load_path)
        self.q_network.load_state_dict(state_dict)
