import csv
import os.path

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.geo import load_graph
from common.utils import haversine, softmax
from baselines.pp_dijkstra import dijkstra_path
from env.rendering import StaticRender

max_step = 100


class CityEnvironment(gym.Env):
    def __init__(self, graph, p, end_node, c=100.0, eta=0.0):
        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)

        self.cur_node_idx = None
        self.start_node_idx = None
        self.end_node_idx = self.nodes.index(end_node)
        self.nodes_idx = list(range(len(self.nodes)))
        self.nodes_idx.pop(self.end_node_idx)
        print('End node: ', end_node)

        self.c = c
        self.eta = eta

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

    def get_mask(self):
        return {i: len(self.p[node]) for i, node in enumerate(self.nodes)}

    def reset(self, render=False, **kwargs):
        node_idx = self.nodes_idx.pop(0)
        self.start_node_idx = node_idx
        self.nodes_idx.append(node_idx)
        self.cur_node_idx = self.start_node_idx

        if render and self.cv_render is None:
            self.cv_render = StaticRender(self.graph)
        return self.__get_state()

    def __get_state(self):
        return self.cur_node_idx

    def step(self, action, is_test=False):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        if action >= len(next_nodes):
            return self.__get_state(), -100.0, False

        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.end_node_idx)

        reward = -self.graph[current_node][next_node][0]['length']  # 米
        if not is_test:
            nodes = self.graph.nodes
            dist1 = -haversine(nodes[current_node], nodes[self.end_node]) * self.eta  # 千米
            dist2 = -haversine(nodes[next_node], nodes[self.end_node]) * self.eta
            if self.eta >= -1000.0:
                reward += 0.99 * dist2 - dist1
                if done: reward += self.c - dist1
            else:
                reward = dist2 / self.eta
                if done: reward += self.c

        self.cur_node_idx = next_node_idx
        return self.__get_state(), reward, done

    def render(self, *args, **kwargs):
        if self.cv_render is None: return
        self.cv_render.draw(*args, **kwargs)


class PolicyIterationTrainer:
    def __init__(self, env, num_actions,
                 gamma=0.99,
                 eval_iter=-1,
                 **kwargs):
        print('Algorithm: PI')
        self.gamma = gamma
        self.env = env
        self.mask = env.get_mask()
        self.e_iter = int(1e4) if eval_iter <= 0 else eval_iter

        num_obs = self.num_obs = len(env.nodes)
        num_act = self.num_act = num_actions

        self.p = np.zeros([num_act, num_obs, num_obs], dtype=np.float32)  # P(s'|s,a)
        self.r = np.zeros([num_act, num_obs, num_obs], dtype=np.float32)  # R(s,a,s')
        for s in range(num_obs):
            for a in range(num_act):
                self.env.cur_node_idx = s
                s_prime, reward, *_ = self.env.step(a)
                if s == s_prime: continue

                self.r[a, s, s_prime] = reward
                self.p[a, s, s_prime] = 1.0

        self.pi = np.zeros((num_obs, num_act))  # pi(s)
        self.v = np.zeros((num_obs,))           # V(s)
        self.q = np.zeros((num_obs, num_act))   # Q(s,a)
        self.pi_seq = [self.pi.copy(), ]
        self.q_seq = [self.q.copy(), ]
        self.v_seq = [self.v.copy(), ]

    def choose_action(self, state):
        act_prob = self.pi[state]
        acts = np.argwhere(act_prob == act_prob.max())
        return np.random.choice(acts.squeeze(axis=1))

    def update(self):
        self.__evaluation()
        ok = self.__improvement()
        self.pi_seq.append(self.pi.copy())
        self.q_seq.append(self.q.copy())
        self.v_seq.append(self.v.copy())
        return ok

    def compute_v(self, pi=None, num_iters=int(1e2)):
        if pi is None: pi = self.pi

        v = self.v.copy()
        for _ in range(num_iters):
            new_v = v.copy()
            for s in range(self.num_obs):
                q = []
                for a in range(self.num_act):
                    q.append(np.dot(self.p[a, s, :],
                                    self.r[a, s, :] + self.gamma * v))
                new_v[s] = np.dot(pi[s], np.array(q))
            # if np.all(np.equal(new_v, v)): break
            v = new_v.copy()
        return v

    def __evaluation(self):
        for _ in range(self.e_iter):
            new_v = self.v.copy()
            for s in range(self.num_obs):
                q = []
                for a in range(self.num_act):
                    q.append(np.dot(self.p[a, s, :],
                                    self.r[a, s, :] + self.gamma * self.v))
                q = np.array(q)
                self.q[s, :] = q[:]
                new_v[s] = np.dot(self.pi[s], q)
            # if np.all(np.equal(new_v, self.v)): break
            self.v = new_v.copy()

    def __improvement(self):
        new_policy = np.zeros_like(self.pi)
        for s in range(self.num_obs):
            q = self.q[s, :]
            q_ = q[:self.mask[s]]
            ids = np.argwhere(q_ == q_.max()).squeeze(axis=1)
            new_policy[s, ids] = 1.0 / len(ids)
        # if np.all(np.equal(new_policy, self.pi)): return False
        self.pi = new_policy
        return True


def train_pi(graph,
             p, num_action,
             center_node,
             eta=0.0,
             c=100.0,
             num_episodes=int(1e3),
             radius=0):
    env = CityEnvironment(graph, p, center_node, eta=eta, c=c)
    trainer = PolicyIterationTrainer(env, num_action, eval_iter=int(1e0))

    episode = None
    desc = '{:>+7.1f}, {:>+7.1f} {:>+7.1f}, Training ...'.format(radius, eta, c)
    for episode in tqdm(range(num_episodes), desc=desc):
        if not trainer.update(): break

    root = 'trained/exp_{}/'.format(radius)
    if not os.path.exists(root): os.mkdir(root)

    test_pi(env, trainer,
            root=root,
            num_episodes=int(1e3),
            epoch=[radius, eta, c, episode])
    np.save(root+'v_rs_{}({})'.format(round(eta, 1), round(c, 1)),
            np.array(trainer.v_seq))

    if eta != 0.0: return

    nodes = env.graph.nodes
    end_node = nodes[env.end_node]
    arr = []
    for i, v1 in enumerate(trainer.v):
        cur_node = nodes[env.nodes[i]]
        v2 = -haversine(cur_node, end_node, km=True)
        arr.append([v1, v2])
    arr = np.array(arr)
    np.save(root+'v_d_{}_{}'.format(c, radius), arr)
    # arr[:, 1] *= 2.0
    print(evaluate_similarity(arr[:, 0], arr[:, 1]))
    arr[:, 0] = (arr[:, 0] - np.mean(arr[:, 0])) / np.std(arr[:, 0])
    arr[:, 1] = (arr[:, 1] - np.mean(arr[:, 1])) / np.std(arr[:, 1])
    print(evaluate_similarity(arr[:, 0], arr[:, 1]))

    return
    row, column = 3, 3

    fig, axes = plt.subplots(row, column)
    for i in range(row):
        for j in range(column):
            idx = i * 3 + j
            min_x, max_x = 100*idx, min(100*(idx+1), len(arr))
            x_list = list(range(min_x, max_x))
            axes[i, j].plot(x_list, arr[min_x:max_x, 0], label='$V^*(s;R_1)$')
            axes[i, j].plot(x_list, -arr[min_x:max_x, 1], label='$-R_2=d(s,s_g)$')
            axes[i, j].legend(loc='upper right')
    plt.show()


def evaluate_similarity(v_values, d_values):
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LinearRegression

    # Pearson 相关系数
    pearson_corr, _ = pearsonr(v_values, d_values)
    # Cosine 相似度
    cosine_sim = cosine_similarity(v_values.reshape(1, -1),
                                   d_values.reshape(1, -1))[0, 0]
    # 线性拟合后残差 MSE
    model = LinearRegression().fit(d_values.reshape(-1, 1), v_values)
    v_predict = model.predict(d_values.reshape(-1, 1))
    mse = np.mean((v_values - v_predict) ** 2)

    return {
        'pearson_corr': pearson_corr,
        'cosine_similarity': cosine_sim,
        'mse_after_linear_fit': mse,
        'linear_fit_slope': model.coef_[0],
        'linear_fit_intercept': model.intercept_
    }


def test_pi(env, trainer, num_episodes=int(1e3), epoch=None, root=None):
    sr_stats, gap_stats = [], []
    for episode in range(num_episodes):
        done_rl, state = False, env.reset()

        # Reinforcement Learning
        total_reward, episode_step = 0.0, 0
        path_rl = [env.cur_node, ]
        while not done_rl:
            action = trainer.choose_action(state)
            next_state, reward, done_rl, *_ = env.step(action, is_test=True)
            path_rl.append(env.cur_node)
            total_reward += reward
            state = next_state
            episode_step += 1
            if episode_step >= max_step: break
        cost_rl = -total_reward

        # Dijkstra
        start_node, end_node = env.start_node, env.end_node
        cost_di, path_di = dijkstra_path(env.graph, start_node, end_node, weight_key='length')
        done_di = path_di[0] == start_node and path_di[-1] == end_node

        if done_rl and done_di:
            gap = (cost_rl - cost_di) / cost_di
            gap_stats.append([gap, int(gap == 0)])
        sr_stats.append([int(done_rl), int(done_di)])

    if len(gap_stats) > 0:
        gap_stats_arr = np.array(gap_stats)
        mean_sr = np.array(sr_stats).mean(0)

        min_gap = gap_stats_arr.min(0)
        mean_gap = gap_stats_arr.mean(0)
        max_gap = gap_stats_arr.max(0)
        result = epoch + [min_gap[0], max_gap[0]] + list(mean_gap) + list(mean_sr)
    else:
        result = epoch + [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    print(result)
    with open(root+'result_dpp_pi.csv', 'a+', newline='') as f:
        csv.writer(f).writerow(result)


def main():
    """
    1e4:   495 nodes and  1148 edges;
    1e5: 19925 nodes and 47408 edges
    """
    np.random.seed(1234)

    radius = 3e4
    place_name = "Nanjing, China"
    args = load_graph(place_name,
                      network_type='drive',
                      center=0,
                      radius=radius,
                      remove=True,
                      render=False)
    # train_pi(*args, eta=0.0, c=5e3, num_episodes=int(1e3), radius=int(radius))

    for c in np.linspace(3.5e3, 5e3, 11):
        for eta in np.linspace(0.0, 0.0, 1):
            train_pi(*args, eta, c, num_episodes=int(1e3), radius=int(radius))


def main2():
    num_episodes = int(1e3)
    test_rate = 1
    place_name = "Nanjing, China"
    graph, p, num_action = load_graph(place_name,
                                      network_type='drive',
                                      center=0,
                                      radius=1e4,
                                      remove=True)
    c = 50.0
    trainers = []
    for eta in np.linspace(-1500.0, 1500.0, 5):
        env = CityEnvironment(graph, p, eta=eta, c=c)
        trainer = PolicyIterationTrainer(env, num_action, eval_iter=1)
        trainers.append([eta, c, trainer])
    lst = [[] for _ in trainers]

    trainer0 = trainers.pop(0)[-1]
    for episode in tqdm(range(num_episodes)):
        trainer0.update()
        for i, [eta, c, trainer] in enumerate(trainers):
            trainer.update()
            # if episode % test_rate == 0:
            #     test_pi(trainer, num_episodes=int(3e2), epoch=[eta, c, episode])

        if episode % test_rate == 0:
            for i, [_, _, trainer] in enumerate(trainers):
                v = trainer0.compute_v(trainer.pi)
                lst[i].append(v)
            lst[-1].append(trainer0.compute_v())

    for i, [eta, c, _] in enumerate(trainers):
        file_name = 'trained/c_2e4/v_rs_{}({})'.format(round(eta, 1), round(c, 1))
        arr = np.array(lst[i])
        print(arr.shape)
        np.save(file_name, arr)


if __name__ == '__main__':
    main()
