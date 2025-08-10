import os
import time
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from common.geo import load_graph


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, emb_dim=256, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(state_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x1 = self.embedding(x[:, 0].to(torch.int32))
        x2 = self.embedding(x[:, 1].to(torch.int32))
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Critic(nn.Module):
    def __init__(self, state_dim, emb_dim=256, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(state_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x1 = self.embedding(x[:, 0].to(torch.int32))
        x2 = self.embedding(x[:, 1].to(torch.int32))
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TrajectoryBuffer:
    def __init__(self, gamma=0.99, lamb=0.95):
        self.gamma = gamma
        self.lamb = lamb

        self.states, self.actions = None, None
        self.rewards, self.values = None, None
        self.log_probs, self.dones = None, None

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_ret_and_adv(self, last_value):
        gae = 0
        advantages, returns = [], []
        for t in reversed(range(len(self.rewards))):
            v = last_value if t == len(self.rewards) - 1 else self.values[t + 1]

            delta = self.rewards[t] + self.gamma * (1 - self.dones[t]) * v - self.values[t]
            gae = delta + self.gamma * self.lamb * (1 - self.dones[t]) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        returns = torch.vstack(returns).detach()
        advantages = torch.vstack(advantages).detach()
        return returns, advantages


class GraphEnvironment:
    def __init__(self, graph, p):
        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)

        self.cur_node_idx = None
        self.start_node, self.start_node_idx = None, None
        self.end_node, self.end_node_idx = None, None

    def reset(self):
        scenario = np.random.choice(len(self.nodes), 2, replace=False)
        self.start_node_idx, self.end_node_idx, *_ = scenario

        self.cur_node_idx = self.start_node_idx
        self.start_node = self.nodes[self.start_node_idx]
        self.end_node = self.nodes[self.end_node_idx]
        return self.__get_state()

    def __get_state(self):
        return torch.tensor([self.cur_node_idx, self.end_node_idx], dtype=torch.int64)

    def step(self, action):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        if action >= len(next_nodes):
            return self.__get_state(), -10.0, False

        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.end_node_idx)
        reward = -self.graph[current_node][next_node][0]['dynamic_weight']
        reward += int(done) * 1e3

        self.cur_node_idx = next_node_idx
        return self.__get_state(), reward, done


class PPOTrainer:
    def __init__(self, graph, num_action, actor_lr=3e-4, critic_lr=3e-4):
        state_dim = len(graph.nodes)
        self.actor = Actor(state_dim=state_dim, action_dim=num_action)
        self.critic = Critic(state_dim=state_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def choose_action(self, state):
        state_tensor = state.unsqueeze(0)
        probs = Categorical(logits=self.actor(state_tensor))
        action = probs.sample()
        log_prob = probs.log_prob(action)
        value = self.critic(state_tensor)
        return action, log_prob, value

    def update(self, state, buffer, epoch=10):
        with torch.no_grad():
            last_value = self.critic(state.unsqueeze(0))

        returns, advantages = buffer.compute_ret_and_adv(last_value.item())
        states = torch.stack(buffer.states)
        actions = torch.tensor(buffer.actions)
        old_log_probs = torch.tensor(buffer.log_probs)

        losses = []
        for _ in range(epoch):  # PPO epochs
            dist = Categorical(logits=self.actor(states))
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            a_loss = -torch.min(surr1, surr2).mean()

            values = self.critic(states).squeeze()
            c_loss = F.mse_loss(returns, values)
            loss = a_loss + 0.5 * c_loss - 0.01 * entropy

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

            losses.append([a_loss.item(), c_loss.item(), entropy.item()])
        return np.array(losses).mean(axis=-1)

    def save_model(self, save_path):
        torch.save(self.actor.state_dict(), save_path+'_actor.pth')
        torch.save(self.critic.state_dict(), save_path+'_critic.pth')

    def load_mode(self, load_path):
        state_dict = torch.load(load_path+'_actor.pth')
        self.actor.load_state_dict(state_dict)
        state_dict = torch.load(load_path+'_critic.pth')
        self.critic.load_state_dict(state_dict)


def train_ppo(graph, p, num_action,
              max_step=100,
              num_episodes=int(1e3),
              log_folder='logs_dpp_ppo'):
    env = GraphEnvironment(graph, p)
    trainer = PPOTrainer(graph, num_action)
    buffer = TrajectoryBuffer()

    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
        print('Removing all previous files!')
    writer = SummaryWriter(log_dir=log_folder)

    start_time = time.time()
    sr_stats, rew_stats, loss_stats = [], [], []
    for episode in range(1, num_episodes+1):
        done, state = False, env.reset()
        buffer.reset()

        total_reward, episode_step = 0.0, 0
        while not done:
            action, log_prob, value = trainer.choose_action(state)
            next_state, reward, done = env.step(action.item())

            buffer.store(state, action, reward, log_prob, value, done)
            state = next_state
            total_reward += reward
            episode_step += 1
            if episode_step >= max_step:
                break

        sr_stats.append(int(done))
        rew_stats.append(total_reward)
        loss_stats.append(trainer.update(state, buffer))

        if episode % 100 == 0:
            end_time = time.time()
            sr = np.mean(sr_stats)
            mean_rew = np.mean(rew_stats)
            mean_loss = np.mean(loss_stats, axis=-1)
            print("Episode: {}".format(episode), end=' | ')
            print("Mean Rew: {:>+7.2f}".format(mean_rew), end=' | ')
            print("Mean Loss: {:>6.2f}".format(mean_loss[0]), end=' | ')
            print("SR: {:>6.2f}".format(sr), end=' | ')
            print("Time: {:>6.2f}".format(end_time-start_time))

            writer.add_scalar("Reward/Total", total_reward, episode)
            writer.add_scalar("Loss/Total", mean_loss[0], episode)
            writer.add_scalar("Loss/Actor", mean_loss[1], episode)
            writer.add_scalar("Loss/Critic", mean_loss[2], episode)
            writer.add_scalar("Entropy", mean_loss[3], episode)
            writer.add_scalar("Success Rate", sr, episode)

            start_time = end_time
            sr_stats, rew_stats, loss_stats = [], [], []
            trainer.save_model('trained/mode_dqp_ppo')

    writer.close()


def main():
    place_name = "南京航空航天大学(将军路校区)"
    graph, p, num_action = load_graph(place_name)
    train_ppo(graph, p, num_action, num_episodes=int(1e5))


if __name__ == "__main__":
    main()
