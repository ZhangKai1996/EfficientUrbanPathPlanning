import csv

import gym
import numpy as np
from tqdm import tqdm

from common.geo import load_graph
from common.utils import haversine
from baselines.pp_dijkstra import dijkstra_path
from env.rendering import StaticRender

max_step = 100


class CityEnvironment(gym.Env):
    def __init__(self, graph, p):
        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)

        self.cur_node_idx = None
        self.start_node_idx = None
        self.end_node_idx = 0

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

    def reset(self, render=False, **kwargs):
        nodes_idx = list(range(len(self.nodes)))
        self.end_node_idx = 0
        nodes_idx.pop(self.end_node_idx)
        self.start_node_idx = np.random.choice(nodes_idx)
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


class PolicyIterationTrainer:
    def __init__(self, env, num_actions,
                 gamma=0.99,
                 eval_iter=-1,
                 improve_iter=-1,
                 **kwargs):
        print('Algorithm: PI')
        self.gamma = gamma
        self.env = env
        self.e_iter = int(1e4) if eval_iter <= 0 else eval_iter
        self.i_iter = int(1e4) if improve_iter <= 0 else improve_iter

        num_obs = self.num_obs = len(env.nodes)
        num_act = self.num_act = num_actions

        self.p = np.zeros([num_act, num_obs, num_obs], dtype=np.float32)  # P(s'|s,a)
        self.r = np.zeros([num_act, num_obs, num_obs], dtype=np.float32)  # R(s,a,s')
        for s in range(num_obs):
            for a in range(num_act):
                self.env.cur_node_idx = s
                s_prime, reward, *_ = self.env.step(a)
                self.r[a, s, s_prime] = reward
                self.p[a, s, s_prime] = 1.0
        print(self.p.shape, self.r.shape)

        random_actions = np.random.randint(0, num_act, size=(num_obs,))
        self.pi = np.eye(num_act)[random_actions]  # pi(s)
        self.v = np.zeros((num_obs,))              # V(s)
        self.q = np.zeros((num_obs, num_act))      # Q(s,a)

    def choose_action(self, state):
        act_prob = self.pi[state]
        acts = np.argwhere(act_prob == act_prob.max())
        return np.random.choice(acts.squeeze(axis=1))

    def update(self):
        self.__evaluation()
        return self.__improvement()

    def __evaluation(self):
        for _ in tqdm(range(self.e_iter), desc='Evaluating ...'):
            old_v = self.v
            new_v = old_v.copy()
            for s in range(self.num_obs):
                q = []
                for a in range(self.num_act):
                    q.append(np.dot(self.p[a, s, :],
                                    self.r[a, s, :] + self.gamma * old_v))
                new_v[s] = np.dot(self.pi[s], np.array(q))
            self.v = new_v.copy()

    def __improvement(self):
        new_policy = np.zeros_like(self.pi)
        for s in range(self.num_obs):
            q = np.array([np.dot(self.p[a, s, :],
                                 self.r[a, s, :] + self.gamma * self.v)
                          for a in range(self.num_act)])
            self.q[s, :] = q[:]
            idx = np.argmax(q)
            new_policy[s, idx] = 1.0
            # ids = np.argwhere(q == q.max()).squeeze(axis=1)
            # for idx in ids:
            #     new_policy[s, idx] = 1.0 / len(ids)

        if np.all(np.equal(new_policy, self.pi)): return False
        self.pi = new_policy
        return True


def train_pi(graph, p, num_action, num_episodes=int(1e3), test_rate=1):
    env = CityEnvironment(graph, p)
    trainer = PolicyIterationTrainer(env, num_action)

    for episode in range(num_episodes):
        trainer.update()
        if episode % test_rate == 0:
            test_pi(env, trainer, num_episodes=int(1e3), epoch=episode)


def test_pi(env, trainer, num_episodes=int(1e3), epoch=0):
    sr_stats, gap_stats = [], []
    for episode in range(num_episodes):
        done_rl, state = False, env.reset()
        print(episode, env.start_node, env.end_node)

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
        print('\t RL: ', done_rl, round(cost_rl, 2), len(path_rl), path_rl)

        # Dijkstra
        start_node, end_node = env.start_node, env.end_node
        cost_di, path_di = dijkstra_path(env.graph, start_node, end_node)
        done_di = path_di[0] == start_node and path_di[-1] == end_node
        print('\t DI: ', done_di, round(cost_di, 2), len(path_di), path_di)

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
    with open('trained/result_dpp_pi_0.csv', 'a+', newline='') as f:
        csv.writer(f).writerow(result)


def main():
    place_name = "南京航空航天大学(将军路校区)"
    graph, p, num_action = load_graph(place_name)
    train_pi(graph, p, num_action, num_episodes=int(1e3))


if __name__ == '__main__':
    main()
