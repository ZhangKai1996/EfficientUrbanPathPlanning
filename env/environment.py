import numpy as np

from common.utils import haversine, softmax
from .rendering import StaticRender


class CityEnvironment:
    def __init__(self, graph, p, num_action, end_node=None, c=100.0, eta=0.0, alpha=1.0):
        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_action = num_action

        self.cur_node_idx = None
        self.start_node_idx = None
        self.end_node_idx = None
        if end_node is not None:
            self.end_node_idx = self.nodes.index(end_node)

        self.c = c
        self.eta = eta
        self.alpha = alpha

        self.rank_value = None
        self.rank_key = None
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

    def get_rank(self, end_node_idx, k=None):
        if self.rank_value is not None: return
        end_node = self.nodes[end_node_idx]

        ret = {}
        for node_idx in range(len(self.nodes)):
            if node_idx == end_node_idx:
                continue
            u = self.graph.nodes[self.nodes[node_idx]]
            v = self.graph.nodes[end_node]
            ret[node_idx] = haversine(u, v)

        sorted_items = sorted(ret.items(), key=lambda x: x[1])
        # 提取排序后的值列表和键列表
        if k is None:
            dist_arr = np.array([item[1] for item in sorted_items])
            self.rank_key = [item[0] for item in sorted_items]
        else:
            dist_arr = np.array([item[1] for item in sorted_items][:k])
            self.rank_key = [item[0] for item in sorted_items][:k]
        self.rank_value = softmax(dist_arr)

    def reset(self, reuse=False, forced=False, render=False):
        """
        city-level: 29650,
        district-level: 6244,
        street-level: 215
        """
        # scenario = np.random.choice(len(self.nodes), 2, replace=False)
        # self.start_node_idx, self.end_node_idx = scenario

        if self.start_node_idx is None: reuse = False
        if not reuse:
            self.get_rank(end_node_idx=self.end_node_idx)
            node_idx = self.rank_key.pop(0)
            self.rank_key.append(node_idx)
            # node_idx = np.random.choice(self.rank_key)
            # node_idx = self.rank_key[-1]
            self.start_node_idx = node_idx

        # # node_idx = self.rank_key.pop(0)
        # # self.rank_key.append(node_idx)
        # if (self.start_node_idx is None
        #         or self.cur_node_idx == self.end_node_idx
        #         or forced):
        #     node_idx = self.rank_key[-1]
        #     # node_idx = np.random.choice(self.rank_key)
        #     self.start_node_idx = node_idx
        # else:
        #     self.start_node_idx = self.cur_node_idx

        self.cur_node_idx = self.start_node_idx
        if render and self.cv_render is None:
            self.cv_render = StaticRender(self.graph)
        return self.__get_state()

    def __get_state(self):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        mask = np.zeros((self.num_action,))
        mask[len(next_nodes):] = -np.inf
        return [self.end_node_idx, self.cur_node_idx, ], mask

    def step(self, action, is_test=False):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.end_node_idx)

        # Reward (COST)
        reward = -self.graph[current_node][next_node][0]['length'] * self.alpha  # 米
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
