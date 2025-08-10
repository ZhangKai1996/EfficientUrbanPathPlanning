import os.path
import time
import shutil

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from algo.memory import ScalableReplayBuffer
from algo.misc import soft_update, LinearSchedule
from algo.model import RNN
from common.geo import load_graph
from common.utils import haversine
from baselines.tsp_held_karp import held_karp

max_step = 100


class TSPSolverGCN:
    def __init__(self, graph,
                 max_num_tasks,
                 tau=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 batch_size=32,
                 epsilon_decay=int(1e6),
                 buffer_size=int(1e6),
                 log_folder='trained/logs_tsp_rnn',
                 has_writer=True):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.batch_size = batch_size

        self.start_node_idx = None
        self.cur_node_idx = None
        self.end_node_idx = None

        self.task_nodes = None
        self.visited = None
        self.matrix = None

        self.max_num_tasks = max_num_tasks
        self.state_dim = 32

        self.q_network = RNN(input_dim=self.state_dim, output_dim=1)
        self.target_network = RNN(input_dim=self.state_dim, output_dim=1)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.tau = tau
        self.gamma = gamma
        self.replay_buffer = ScalableReplayBuffer(capacity=buffer_size)
        self.schedule = LinearSchedule(epsilon_decay, epsilon_end, epsilon_start)

        self.writer = None
        if has_writer:
            if os.path.exists(log_folder):
                shutil.rmtree(log_folder)
                print('Removing all previous files!')
            self.writer = SummaryWriter(log_dir=log_folder)

    @property
    def start_node(self):
        return self.nodes[self.start_node_idx]

    @property
    def end_node(self):
        return self.nodes[self.end_node_idx]

    def get_all_nodes(self):
        return [self.start_node_idx, ] + self.task_nodes + [self.end_node_idx, ]

    def __get_state(self):
        all_nodes = self.get_all_nodes()
        cur_node_arr = list(format(self.cur_node_idx, '010b'))
        end_node_arr = list(format(self.end_node_idx, '010b'))
        i1 = all_nodes.index(self.cur_node_idx)

        state = []
        for i, task_node_idx in enumerate(self.task_nodes):
            if self.visited[task_node_idx] == 1:
                continue

            arr1 = list(format(task_node_idx, '010b'))
            arr2 = cur_node_arr[:] + [self.matrix[i1][i+1], ]
            arr3 = end_node_arr[:] + [self.matrix[i+1][-1], ]
            state_ = np.concatenate([np.array(arr1, dtype=np.float32),
                                     np.array(arr2, dtype=np.float32),
                                     np.array(arr3, dtype=np.float32)])
            state.append(state_)
        if len(state) <= 0: state = [np.zeros(32, )]
        return np.vstack(state)

    def reset(self):
        num_tasks = np.random.randint(2, self.max_num_tasks + 1)
        node_indexes = list(range(len(self.nodes)))
        self.task_nodes = np.random.choice(node_indexes[1:-1], num_tasks, replace=False)
        self.task_nodes = sorted(self.task_nodes)

        self.visited = {node: 0 for node in self.task_nodes}
        self.matrix = np.random.uniform(low=0.5, high=1.5,
                                        size=(num_tasks+2, num_tasks+2))

        self.start_node_idx = 0
        self.end_node_idx = len(self.nodes) - 1
        self.cur_node_idx = self.start_node_idx
        return self.__get_state()

    def choose_action(self, state, t=None):
        if t is not None:
            if np.random.uniform() < self.schedule.value(t):
                return np.random.choice(len(state))

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state_tensor).squeeze(0).detach().numpy()
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def step(self, action):
        unvisited_task_nodes = [v for v in self.task_nodes if self.visited[v] == 0]
        next_node_idx = unvisited_task_nodes[action]

        nodes = self.graph.nodes
        cur_node = self.nodes[self.cur_node_idx]
        next_node = self.nodes[next_node_idx]
        dist = haversine(nodes[cur_node], nodes[next_node])

        all_nodes = self.get_all_nodes()
        i1 = all_nodes.index(self.cur_node_idx)
        i2 = all_nodes.index(next_node_idx)
        coff = self.matrix[i1][i2]
        reward = -dist * coff

        self.cur_node_idx = next_node_idx
        self.visited[next_node_idx] = 1
        done = sum(self.visited.values()) == len(self.task_nodes)
        if done:
            i3 = all_nodes.index(self.end_node_idx)
            coff = self.matrix[i2][i3]
            end_node = self.nodes[self.end_node_idx]
            reward -= haversine(nodes[next_node], nodes[end_node]) * coff
        return self.__get_state(), reward, done

    def update_q_network(self, t):
        if len(self.replay_buffer) < int(1e4): return None
        if t % 100 != 0: return None

        # Sample a batch of experiences from the buffer
        losses = []
        batch_dict = self.replay_buffer.sample(self.batch_size, num_keys=8)
        total_gradients = None
        for key, batch in batch_dict.items():
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64)
            rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32)
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32)

            # Compute Q values using current Q network
            q_values = self.q_network(states_tensor)
            next_q_values = self.target_network(next_states_tensor)
            next_q_values_max = next_q_values.max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values_max * (1 - dones_tensor)
            # Get the Q values for the actions taken
            action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            # Compute loss
            loss = F.mse_loss(action_q_values, target_q_values.detach())

            # Back propagate to compute gradients
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
            # Accumulate gradients
            if total_gradients is None:
                total_gradients = {name: param.grad.clone() for name, param in self.q_network.named_parameters()}
            else:
                for name, param in self.q_network.named_parameters():
                    total_gradients[name] += param.grad
            # Store individual task loss for later analysis
            losses.append(loss.item())

        # After all tasks, average the gradients (if you want to average them)
        for name, param in self.q_network.named_parameters():
            param.grad = total_gradients[name] / len(batch_dict)
        # Update the model parameters with the averaged gradients
        self.optimizer.step()

        if t % 100 == 0:
            soft_update(self.target_network, self.q_network, self.tau)
        return np.mean(losses)

    def save_model(self, save_path):
        torch.save(self.q_network.state_dict(), save_path)

    def load_mode(self, load_path):
        state_dict = torch.load(load_path)
        self.q_network.load_state_dict(state_dict)

    def train(self, num_episodes=1000):
        rew_stats, sr_stats = [], []
        step_stats, loss_stats = [], []
        step = 0

        start_time = time.time()
        for episode in range(1, num_episodes + 1):
            done, state = False, self.reset()
            total_reward, episode_step = 0.0, 0
            while not done:
                action = self.choose_action(state, t=step)
                next_state, reward, done, *_ = self.step(action)
                # print(state.shape, action, next_state.shape, reward, done)

                self.replay_buffer.push(state, action, reward, next_state, float(done))
                # Update Q network
                loss = self.update_q_network(t=step)
                if loss is not None:
                    loss_stats.append(loss)

                state = next_state
                total_reward += reward
                episode_step += 1
                step += 1
                if episode_step >= max_step: break

            step_stats.append(episode_step)
            rew_stats.append(total_reward)
            sr_stats.append(int(done))
            if episode % 100 == 0:
                end_time = time.time()
                mean_step = np.mean(step_stats)
                mean_rew = np.mean(rew_stats)
                mean_loss = np.mean(loss_stats)
                sr = np.mean(sr_stats)
                eps = self.schedule.value(step)

                print("Episode: {}".format(episode), end=', ')
                print("Step: {}".format(step), end=', ')
                print("Mean Step: {:>7.2f}".format(mean_step), end=', ')
                print("Mean Rew: {:>7.2f}".format(mean_rew), end=', ')
                print("SR: {:>6.2f}".format(sr), end=', ')
                print("Epsilon: {:>6.3f}".format(eps), end=',')
                print("Time: {:>5.2f}".format(end_time-start_time))

                if self.writer is not None:
                    self.writer.add_scalar('Mean Step', mean_step, episode)
                    self.writer.add_scalar('Mean Rew', mean_rew, episode)
                    self.writer.add_scalar('Mean Loss', mean_loss, episode)
                    self.writer.add_scalar('SR', sr, episode)
                    self.writer.add_scalar('Epsilon', eps, step)

                rew_stats, sr_stats = [], []
                step_stats, loss_stats = [], []
                start_time = end_time
                self.save_model('trained/model_tsp_rnn.pth')

    def test(self, num_episodes=1000, name='fig'):
        self.load_mode('trained/model_tsp_rnn.pth')

        stats, gap_stats = [], []
        for ep in range(num_episodes):
            done_rl, state = False, self.reset()
            print('\nEpisode: {}'.format(ep), end=', ')
            print('Start: {}, End: {}'.format(self.start_node, self.end_node), end=', ')
            print('Task nodes: ', [self.nodes[u] for u in self.task_nodes])

            # Reinforcement Learning
            total_reward, episode_step = 0.0, 0
            path_rl = [self.nodes[self.cur_node_idx], ]
            while not done_rl:
                action = self.choose_action(state)
                next_state, reward, done_rl = self.step(action)
                # print('\t', state, action, next_state, reward, done_rl)
                path_rl.append(self.nodes[self.cur_node_idx])
                total_reward += reward
                state = next_state
                episode_step += 1
                if episode_step >= max_step: break
            # print(total_reward, done_rl)
            cost_rl = -total_reward
            print("\t Optimal path (RL):", round(cost_rl, 2), done_rl, path_rl)

            # Dijkstra
            nodes = self.graph.nodes
            all_nodes = self.get_all_nodes()
            all_nodes = [self.nodes[node] for node in all_nodes]

            dist_matrix = np.zeros((len(all_nodes), len(all_nodes)))
            for i, u in enumerate(all_nodes):
                for j, v in enumerate(all_nodes):
                    if i == j: continue
                    coff = self.matrix[i][j]
                    dist_matrix[i][j] = haversine(nodes[u], nodes[v]) * coff

            cost_dp, path_dp = held_karp(dist_matrix=dist_matrix)
            path_dp = [all_nodes[node] for node in path_dp]
            done_dp = path_dp[0] == self.start_node and path_dp[-1] == self.end_node
            print("\t Optimal path (DP):", round(cost_dp, 2), done_dp, path_dp)

            stats.append([int(done_rl), cost_rl, int(done_dp), cost_dp])
            if done_rl and done_dp:
                gap_stats.append((cost_rl-cost_dp)/cost_dp)

        fig, axes = plt.subplots(1, 2, constrained_layout=True)

        mean_values = np.array(stats).mean(axis=0)
        axes[0].hist(gap_stats, bins=20, alpha=0.6, density=True)
        title = 'SR(rl):{:>6.3f}, '.format(mean_values[0])
        title += 'SR(di):{:>6.3f}'.format(mean_values[2])
        axes[0].set_title(title)

        axes[1].boxplot(gap_stats, notch=True, showmeans=True)
        title = 'GAP:{:>6.3f}'.format(np.mean(gap_stats))
        axes[1].set_title(title)

        plt.savefig(name+'.png', dpi=300)
        # plt.show()
        plt.close()


def main():
    place_name = "南京航空航天大学(将军路校区)"
    max_num_tasks = 30
    num_episodes = int(1e6)
    eps_decay = int(max_num_tasks * num_episodes * 0.2)

    graph, *_ = load_graph(place_name)
    planner = TSPSolverGCN(graph, max_num_tasks, epsilon_decay=eps_decay)
    planner.train(num_episodes=num_episodes)


if __name__ == '__main__':
    main()
