import csv
import os.path
import time
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from algo.memory import ScalableReplayBuffer
from algo.misc import soft_update, LinearSchedule, compute_entropy
from baselines.pp_dijkstra import dijkstra_path
from common.geo import load_graph
from common.utils import haversine
from env.rendering import StaticRender

max_step = 100


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, emb_dim=256, hidden_dim=64):
        super(DuelingDQN, self).__init__()
        self.embedding = nn.Embedding(state_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Value 分支
        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)
        # Advantage 分支
        self.advantage_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.advantage_fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x1 = self.embedding(state[:, 0])
        x2 = self.embedding(state[:, 1])
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)  # [batch_size, 1]
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)  # [batch_size, action_dim]
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class CityEnvironment:
    def __init__(self, graph, p):
        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)

        self.cur_node_idx = None
        self.start_node_idx = None
        self.end_node_idx = None

        self.cv_render = None

    @property
    def start_node(self):
        return self.nodes[self.start_node_idx]

    @property
    def end_node(self):
        return self.nodes[self.end_node_idx]

    @property
    def cur_node(self):
        return self.nodes[self.cur_node_idx]

    def reset(self, render=False):
        scenario = np.random.choice(len(self.nodes), 2, replace=False)
        self.start_node_idx, self.end_node_idx = scenario

        # self.end_node_idx = 0
        # nodes_idx = list(range(len(self.nodes)))[1:]
        # self.start_node_idx = np.random.choice(nodes_idx)

        self.cur_node_idx = self.start_node_idx

        if render and self.cv_render is None:
            self.cv_render = StaticRender(self.graph)
        return self.__get_state()

    def __get_state(self):
        return [self.end_node_idx, self.cur_node_idx, ]

    def step(self, action, is_test=False):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        if action >= len(next_nodes):
            return self.__get_state(), -10.0, False

        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.end_node_idx)
        reward = -self.graph[current_node][next_node][0]['dynamic_weight']
        if not is_test:
            # nodes = self.graph.nodes
            # dist1 = -haversine(nodes[current_node], nodes[self.end_node])
            # dist2 = -haversine(nodes[next_node], nodes[self.end_node])
            # reward += (0.99 * dist2 - dist1) * 100.0
            # reward = dist2
            reward += int(done) * 1e2

        self.cur_node_idx = next_node_idx
        return self.__get_state(), reward, done

    def render(self, *args, **kwargs):
        if self.cv_render is None: return
        self.cv_render.draw(*args, **kwargs)


class DQNTrainer:
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

    def choose_action(self, state, t=None):
        if t is not None:
            if np.random.uniform() < self.schedule.value(t):
                return np.random.choice(self.num_action)

        state_tensor = torch.tensor(state, dtype=torch.int32).unsqueeze(0)
        q_values = self.q_network(state_tensor)[0].detach().numpy()
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def update_q_network(self, t, update_rate=100):
        if t < int(1e4) or t % 10 != 0: return None, None, None

        # Sample a batch of experiences from the buffer
        batch = self.replay_buffer.sample_(self.batch_size, num_keys=10)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.int32)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.int32)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32)

        # 当前 Q 值
        q_values = self.q_network(states_tensor)
        action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Double DQN 目标
        next_q_values_online = self.q_network(next_states_tensor)
        next_actions = next_q_values_online.argmax(1, keepdim=True)
        next_q_values_target = self.target_network(next_states_tensor)
        next_q_values_selected = next_q_values_target.gather(1, next_actions).squeeze(1)

        # Reward Shaping：shaped_rewards = r + gamma * \Phi(s') - \Phi(s)
        eta = 0.1
        phi_state = action_q_values * eta
        phi_next_state = next_q_values_selected * eta
        shaped_rewards = rewards_tensor + self.gamma * phi_next_state.detach() - phi_state.detach()

        # Q-learning loss
        target_q_values = shaped_rewards + self.gamma * next_q_values_selected * (1 - dones_tensor)
        # target_q_values = rewards_tensor + self.gamma * next_q_values_selected * (1 - dones_tensor)
        loss = F.mse_loss(action_q_values, target_q_values.detach())
        entropy = compute_entropy(q_values)

        # Optimize Q network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        if t % update_rate == 0:
            soft_update(self.target_network, self.q_network, self.tau)
        return loss.item(), entropy.item()

    def save_model(self, save_path):
        torch.save(self.q_network.state_dict(), save_path)

    def load_mode(self, load_path):
        state_dict = torch.load(load_path)
        self.q_network.load_state_dict(state_dict)


def train_dqn(graph, p,
              num_action,
              num_episodes=int(1e3),
              save_rate=100,
              test_rate=5000,
              suffix='dpp_dqn_rs'):
    env = CityEnvironment(graph, p)
    trainer = DQNTrainer(graph, p, num_action, epsilon_decay=int(1e6))

    log_folder = 'trained/logs_' + suffix
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
        print('Removing all previous files!')
    writer = SummaryWriter(log_dir=log_folder)

    rew_stats, sr_stats, loss_stats = [], [], []
    step, start_time = 0, time.time()
    for episode in range(1, num_episodes + 1):
        done, state = False, env.reset()

        total_reward, episode_step = 0.0, 0
        while not done:
            action = trainer.choose_action(state, t=step)
            next_state, reward, done, *_ = env.step(action)
            experience = (state, action, reward, next_state, float(done))
            trainer.replay_buffer.push(*experience, key=env.end_node_idx)
            # Update Q network
            losses = trainer.update_q_network(t=step)
            if None not in losses: loss_stats.append(losses)

            state = next_state
            total_reward += reward
            episode_step += 1
            step += 1
            if episode_step >= max_step: break

        rew_stats.append(total_reward)
        sr_stats.append(int(done))
        if episode % save_rate == 0:
            end_time = time.time()
            mean_rew = np.mean(rew_stats)
            sr = np.mean(sr_stats)
            eps = trainer.epsilon(step)

            print("Episode: {}".format(episode), end=', ')
            print("Step: {}".format(step), end=', ')
            print("Mean Rew: {:>7.2f}".format(mean_rew), end=', ')
            print("SR: {:>6.2f}".format(sr), end=', ')
            print("Epsilon: {:>6.3f}".format(eps), end=',')
            print("Time: {:>5.2f}".format(end_time-start_time))

            if len(loss_stats) > 0:
                mean_loss = np.array(loss_stats).mean(axis=-1)
                writer.add_scalar('Loss/Pi', mean_loss[0], episode)
                writer.add_scalar('Entropy', mean_loss[1], episode)
            writer.add_scalar('Reward', mean_rew, episode)
            writer.add_scalar('Success Rate', sr, episode)
            writer.add_scalar('Epsilon', eps, step)

            if episode % test_rate == 0:
                test_dqn(env, trainer, num_episodes=int(1e3), epoch=episode, suffix=suffix)

            rew_stats, sr_stats, loss_stats = [], [], []
            start_time = end_time
            trainer.save_model('trained/model_{}.pth'.format(suffix))


def test_dqn(env, trainer, num_episodes=int(1e3), epoch=0, suffix='dpp_dqn'):
    trainer.load_mode('trained/model_{}.pth'.format(suffix))

    sr_stats, gap_stats = [], []
    for ep in range(num_episodes):
        done_rl, state = False, env.reset()

        # Reinforcement Learning
        total_reward, episode_step = 0.0, 0
        path_rl = [env.cur_node, ]
        while not done_rl:
            action = trainer.choose_action(state)
            next_state, reward, done_rl = env.step(action, is_test=True)
            path_rl.append(env.cur_node)
            total_reward += reward
            state = next_state
            episode_step += 1
            if episode_step >= max_step: break
        cost_rl = -total_reward

        # Dijkstra
        start_node, end_node = env.start_node, env.end_node
        cost_di, path_di = dijkstra_path(env.graph, start_node, end_node)
        done_di = path_di[0] == start_node and path_di[-1] == end_node

        if done_rl and done_di:
            gap = (cost_rl - cost_di) / cost_di
            gap_stats.append([gap, int(gap == 0)])
        sr_stats.append([int(done_rl), int(done_di)])

    if len(gap_stats) <= 0: return
    gap_stats_arr = np.array(gap_stats)
    mean_sr = np.array(sr_stats).mean(0)

    min_gap = gap_stats_arr.min(0)
    mean_gap = gap_stats_arr.mean(0)
    max_gap = gap_stats_arr.max(0)
    result = [epoch, min_gap[0], max_gap[0]] + list(mean_gap) + list(mean_sr)
    with open('trained/result_{}.csv'.format(suffix), 'a+', newline='') as f:
        csv.writer(f).writerow(result)


def main():
    place_name = "南京航空航天大学(将军路校区)"
    graph, p, num_action = load_graph(place_name)
    train_dqn(graph, p, num_action, num_episodes=int(1e6))


if __name__ == '__main__':
    main()
