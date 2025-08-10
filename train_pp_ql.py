import csv
import math
import os.path
import time
import shutil

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from algo.misc import LinearSchedule
from baselines.pp_dijkstra import dijkstra_path
from common.geo import load_graph
from common.utils import haversine

max_step = 100
suffix = 'pp_ql_test'


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
                 log_folder='trained/logs_{}'.format(suffix),
                 has_writer=True):
        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_action = num_action
        self.batch_size = batch_size

        self.cur_node_idx = None
        self.start_node, self.start_node_idx = None, 0
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

    def __get_state(self):
        return [self.end_node_idx, self.cur_node_idx]

    def epsilon(self, t):
        return self.schedule.value(t)

    def reset(self):
        scenario = np.random.choice(len(self.nodes), 2, replace=False)
        start_node_idx, self.end_node_idx = scenario
        # start_node_idx, self.end_node_idx = 0, 100

        self.cur_node_idx = self.start_node_idx
        self.start_node = self.nodes[self.start_node_idx]
        self.end_node = self.nodes[self.end_node_idx]
        return self.__get_state()

    def choose_action(self, state, t=None):
        if t is not None:
            if np.random.uniform() < self.schedule.value(t):
                return np.random.choice(self.num_action)

        i1 = state[0]
        i2 = state[1]
        q_values = self.q_table[i1][i2]
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def step(self, action, test=False):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        if action >= len(next_nodes):
            return self.__get_state(), -10.0, False

        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.end_node_idx)

        reward = -self.graph[current_node][next_node][0]['dynamic_weight']
        if not test:
            nodes = self.graph.nodes
            dist1 = -haversine(nodes[current_node], nodes[self.end_node])
            dist2 = -haversine(nodes[next_node], nodes[self.end_node])
            reward += (0.99 * dist2 - dist1) * 200.0

            # reward = dist2

            # dist1, *_ = dijkstra_path(self.graph, current_node, self.end_node)
            # dist2, *_ = dijkstra_path(self.graph, next_node, self.end_node)
            # reward += (-0.99 * dist2 + dist1) * 11.0
            # reward += int(done) * 1e2

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

                if episode % 5000 == 0:
                    self.test(num_episodes=int(1e3), epoch=episode)

                rew_stats, sr_stats = [], []
                step_stats, loss_stats = [], []
                start_time = end_time
                self.save_model('trained/model_{}'.format(suffix))

    def test(self, num_episodes=1000, epoch=0):
        self.load_mode('trained/model_{}.npy'.format(suffix))

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

    def random_path(self, name='fig', num_episodes=int(1e4)):
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        records = {'0': [], '1': []}
        for i in tqdm(range(num_episodes), desc='Random Path ...'):
            done, state = False, self.reset()
            total_reward, episode_step = 0.0, 0

            path = [self.start_node, ]
            while not done:
                action = np.random.choice(self.num_action)
                next_state, reward, done, *_ = self.step(action)
                path.append(self.nodes[self.cur_node_idx])
                total_reward += reward
                state = next_state
                episode_step += 1
                if episode_step >= max_step: break

            nodes = self.graph.nodes
            cur_node = self.nodes[self.cur_node_idx]
            phi_s0 = -haversine(nodes[self.start_node], nodes[self.end_node])
            phi_st = -haversine(nodes[cur_node], nodes[self.end_node])
            ef = math.pow(self.gamma, len(path))
            rewards = []
            for times in range(0, 300, 100):
                delta = ef * phi_st * times - phi_s0 * times
                rewards.append(total_reward+delta)

            phi_s0, *_ = dijkstra_path(self.graph, self.start_node, self.end_node)
            phi_st, *_ = dijkstra_path(self.graph, cur_node, self.end_node)
            for times in range(1, 12, 5):
                delta = ef * phi_st*times - phi_s0*times
                rewards.append(total_reward + delta)

            key = '{}'.format(int(done))
            records[key].append([i, episode_step] + rewards)

        record_array_0 = np.array(records['0'])
        record_array_1 = np.array(records['1'])
        print(record_array_0.shape, record_array_1.shape)

        fig, axes = plt.subplots(2, 1)
        count = 0
        if len(record_array_0) > 0:
            length = len(record_array_0[:, 1])
            xs = np.array(list(range(length)))
            count += length
            sorted_arr = record_array_0[record_array_0[:, 2].argsort()]
            for i in range(sorted_arr.shape[-1]):
                if i <= 1: continue
                arr = sorted_arr[:, i]
                label = 'not done {}'.format(i)
                axes[0].plot(xs, arr, label=label, linestyle='dotted', alpha=0.3)
            axes[0].set_title('0: {}, '.format(length))
            axes[0].legend()

        if len(record_array_1) > 0:
            length = len(record_array_1[:, 1])
            xs = np.array(list(range(count, count+length)))
            count += length

            sorted_arr = record_array_1[record_array_1[:, 2].argsort()]
            for i in range(sorted_arr.shape[-1]):
                if i <= 1: continue
                arr = sorted_arr[:, i]
                label = 'done {}'.format(i)
                plt.plot(xs, arr, label=label, linestyle='dotted', alpha=0.3)
            axes[1].set_title('1: {}, '.format(length))

        plt.savefig(name+'.png', dpi=300)
        plt.close()


def main():
    np.random.seed(1234)
    place_name = "南京航空航天大学(将军路校区)"
    # place_name = "Jiangning, Nanjing, China"
    graph, p, num_action = load_graph(place_name, remove=False)
    # planner = PathPlannerQL(graph, p, num_action, epsilon_decay=int(1e6))
    # planner.train(num_episodes=int(1e6))

    planner = PathPlannerQL(graph, p, num_action, has_writer=False)
    planner.random_path('fig_nuaa_1', num_episodes=int(1e4))


if __name__ == '__main__':
    main()
