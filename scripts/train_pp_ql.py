import csv
import os.path
import time
import shutil

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from algo.misc import LinearSchedule
from baselines.pp_dijkstra import dijkstra_path
from common.geo import load_graph, add_dynamic_weight
from common.utils import haversine

max_step = 100


class PathPlannerQL:
    def __init__(self, graph, p,
                 num_action,
                 lr=1e-3,
                 tau=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 batch_size=32,
                 epsilon_decay=int(1e6),
                 log_folder='trained/logs_pp_ql',
                 has_dynamics=False,
                 has_writer=True):
        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_action = num_action
        self.batch_size = batch_size

        self.cur_node_idx = None
        self.start_node = None
        self.end_node, self.end_node_idx = None, None

        self.state_dim = len(self.nodes)
        self.action_dim = num_action
        self.q_table = np.zeros((self.state_dim,
                                 self.state_dim,
                                 self.action_dim), dtype=np.float32)  # Q值表，初始化为0
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.schedule = LinearSchedule(epsilon_decay, epsilon_end, epsilon_start)

        self.writer = None
        if has_writer:
            if os.path.exists(log_folder):
                shutil.rmtree(log_folder)
                print('Removing all previous files!')
            self.writer = SummaryWriter(log_dir=log_folder)
        self.has_dynamics = has_dynamics

    def __get_state(self):
        return [self.end_node_idx, self.cur_node_idx]

    def reset(self):
        # start_node_idx, self.end_node_idx = np.random.choice(len(self.nodes), 2, replace=False)
        # end_nodes = list(range(90, 100))
        start_node_idx = 0
        self.end_node_idx = 100
        # self.end_node_idx = np.random.choice(end_nodes)

        self.cur_node_idx = start_node_idx
        self.start_node = self.nodes[start_node_idx]
        self.end_node = self.nodes[self.end_node_idx]
        add_dynamic_weight(self.graph, dynamic=self.has_dynamics)
        return self.__get_state()

    def choose_action(self, state, t=None):
        if t is not None:
            if np.random.uniform() < self.schedule.value(t):
                return np.random.choice(self.num_action)

        i1 = state[0]
        i2 = state[1]
        q_values = self.q_table[i1][i2]
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        # print(q_values, end=', ')
        return np.random.choice(indexes)

    def step(self, action, test=False):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        if action >= len(next_nodes):
            return self.__get_state(), -10.0, False

        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.end_node_idx)
        if not test:
            # nodes = self.graph.nodes
            # dist = haversine(nodes[next_node], nodes[self.end_node])
            # reward = -dist
            # reward += int(done) * 1e4
            reward = -self.graph[current_node][next_node][0]['dynamic_weight']
            reward += int(done) * 1e5
        else:
            reward = -self.graph[current_node][next_node][0]['dynamic_weight']
            reward += int(done) * 1e2

        self.cur_node_idx = next_node_idx
        return self.__get_state(), reward, done

    def update_q_network(self, state, action, next_state, reward, done):
        i1 = state[0]
        i2 = state[1]
        current_q = self.q_table[i1, i2, action]

        ni1 = next_state[0]
        ni2 = next_state[1]
        next_action = np.argmax(self.q_table[ni1, ni2])  # 获取下一个状态的最大Q值动作
        target_q_next = self.q_table[ni1, ni2, next_action]
        target_q = reward + self.gamma * target_q_next * (1 - int(done))
        td_error = target_q - current_q
        self.q_table[i1, i2, action] += self.lr * td_error
        return td_error

    def save_model(self, save_path):
        np.save(save_path, self.q_table)

    def load_mode(self, load_path):
        self.q_table = np.load(load_path)

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
                # Update Q network
                loss = self.update_q_network(state, action, next_state, reward, float(done))
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
                self.save_model('trained/model_pp_ql')

    def test(self, num_episodes=1000, epoch=0):
        self.load_mode('trained/model_pp_ql.npy')

        sr_stats, gap_stats = [], []
        for ep in range(num_episodes):
            done_rl, state = False, self.reset()
            if self.start_node == self.end_node: continue

            # Reinforcement Learning
            total_reward, episode_step = 0.0, 0
            path_rl = [self.nodes[self.cur_node_idx], ]
            while not done_rl:
                action = self.choose_action(state)
                next_state, reward, done_rl = self.step(action, test=True)
                # print('\t', state, action, next_state, reward, done_rl)
                path_rl.append(self.nodes[self.cur_node_idx])
                total_reward += reward
                state = next_state
                episode_step += 1
                if episode_step >= max_step: break
            # print(total_reward, done_rl)
            cost_rl = -total_reward

            # Dijkstra
            cost_di, path_di = dijkstra_path(self.graph, self.start_node, self.end_node)
            done_di = path_di[0] == self.start_node and path_di[-1] == self.end_node

            if done_rl and done_di:
                gap = (cost_rl - cost_di) / cost_di
                gap_stats.append(gap)
            sr_stats.append([int(done_rl), int(done_di)])

        mean_sr = np.array(sr_stats).mean(0)
        mean_gap = np.mean(gap_stats)
        result = [epoch, mean_gap, ] + list(mean_sr)
        with open('trained/result_pp_ql.csv', 'a+', newline='') as f:
            csv.writer(f).writerow(result)

    def random_path(self, name='fig', num_episodes=int(1e4)):
        records = {'0': [], '1': []}
        for i in range(num_episodes):
            done, state = False, self.reset()
            total_reward, episode_step = 0.0, 0
            while not done:
                action = np.random.choice(self.num_action)
                next_state, reward, done, *_ = self.step(action)
                total_reward += reward
                state = next_state
                episode_step += 1
                if episode_step >= max_step:
                    break

            key = '{}'.format(int(done))
            records[key].append([i, total_reward, episode_step])

        record_array_0 = np.array(records['0'])
        record_array_1 = np.array(records['1'])
        print(record_array_0.shape, record_array_1.shape)

        fig, axes = plt.subplots(2, 1, constrained_layout=True)

        count = 0
        title = ''
        if len(record_array_0) > 0:
            length = len(record_array_0[:, 1])
            xs = np.array(list(range(length)))
            count += length

            axes[0].scatter(xs, np.sort(record_array_0[:, 1]), color='blue', label='prob_0', alpha=0.6)
            title += '0: {}, '.format(length)

        if len(record_array_1) > 0:
            length = len(record_array_1[:, 1])
            xs = np.array(list(range(count, count+length)))
            count += length

            axes[0].scatter(xs, np.sort(record_array_1[:, 1]), color='red', label='prob_1', alpha=0.6)
            title += '1: {}, '.format(length)
            # axes[2].hist(record_array_1[:, 2], bins=20, color='red', label="Data 1", alpha=0.6, density=True)

        [ax.legend() for ax in axes]
        axes[0].set_title(title)

        plt.savefig(name+'.png', dpi=300)
        # plt.show()
        plt.close()


def main():
    place_name = "南京航空航天大学(将军路校区)"
    graph, p, num_action = load_graph(place_name)

    # Step 2: 使用 DQN 来训练路径规划智能体
    planner = PathPlannerQL(graph, p, num_action, epsilon_decay=int(1e6))
    # planner.random_path(name='trained/fig_nuaa', num_episodes=int(2e4))
    # planner.train(num_episodes=int(1e5))

    # planner = PathPlannerQL(graph, p, num_action, has_writer=False)
    planner.test(num_episodes=int(1e4))


if __name__ == '__main__':
    main()
