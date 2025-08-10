import os

import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.geo import load_graph
from common.utils import softmax

from generate_scenarios import parse_config

max_step = 100
_, scenarios = parse_config()


class DQNEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim=256, hidden_dim=64):
        super(DQNEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = self.embedding(x[:, 0].to(torch.int32))
        x2 = self.embedding(x[:, 1].to(torch.int32))
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TopKPathExtractor:
    def __init__(self, graph, p, num_action, load_path):
        self.p = p
        self.graph = graph
        self.num_action = num_action
        self.nodes = list(graph.nodes)

        self.cur_node_idx = None
        self.start_node = None
        self.end_node, self.end_node_idx = None, None

        self.pi_d = DQNEmbedding(len(self.nodes), num_action)
        state_dict = torch.load(load_path)
        self.pi_d.load_state_dict(state_dict)

    def __get_state(self):
        return [self.cur_node_idx, self.end_node_idx]

    def reset(self, start_node_idx, end_node_idx):
        self.end_node_idx = end_node_idx

        self.cur_node_idx = start_node_idx
        self.start_node = self.nodes[start_node_idx]
        self.end_node = self.nodes[self.end_node_idx]
        return self.__get_state()

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.int32).unsqueeze(0)
        q_values = self.pi_d(state_tensor).squeeze(0).detach().numpy()
        probs = softmax(q_values)
        action = np.random.choice(self.num_action, p=probs)
        return action

    def step(self, action):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        if action >= len(next_nodes):
            return self.__get_state(), -10.0, False

        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.end_node_idx)
        reward = -self.graph[current_node][next_node][0]['dynamic_weight']

        self.cur_node_idx = next_node_idx
        return self.__get_state(), reward, done

    def rollout_paths(self, start, end, num_episodes=int(1e3)):
        """
        使用策略 pi_0 从起点出发，rollout 一条路径直到终点
        """
        path_dict, cost_dict = {}, {}
        check_list = []
        for ep in range(num_episodes):
            done_rl, state = False, self.reset(start, end)
            print('\nEpisode: {}, Start: {}, End: {}'.format(ep, self.start_node, self.end_node))

            # Reinforcement Learning
            total_reward, episode_step = 0.0, 0
            path = [self.nodes[self.cur_node_idx], ]
            while not done_rl:
                action = self.choose_action(state)
                next_state, reward, done_rl = self.step(action)
                print('\t\t>>>', state, action, next_state, reward, done_rl)
                path.append(self.nodes[self.cur_node_idx])
                total_reward += reward
                state = next_state
                episode_step += 1
                if episode_step >= max_step: break

            if done_rl:
                path_str = '-'.join([str(v) for v in path])
                if path_str not in check_list:
                    cost_dict[ep] = -total_reward
                    path_dict[ep] = path
                    check_list.append(path_str)

        sorted_items = sorted(cost_dict.items(), key=lambda item: item[1])
        return path_dict, sorted_items

    def extract_subgraph(self, paths):
        node_set = set()
        for path in paths: node_set.update(path)

        subgraph = self.graph.subgraph(node_set).copy()
        edges = []
        for u, v, data in subgraph.edges(data=True):
            key = '{}-{}'.format(u, v)
            edges.append(key)
        return subgraph, node_set, edges


def load_pa_table(num_paths):
    file_path = 'data/pa_{}.yaml'.format(num_paths)
    assert os.path.exists(file_path)
    with open(file_path, 'r') as f:
        atte = yaml.load(f.read(), Loader=yaml.FullLoader)
    return atte


def main():
    # 获取某个地方的图（你可以根据需要改变位置）
    place_name = "南京航空航天大学(将军路校区)"
    graph, p, num_action = load_graph(place_name=place_name)
    extractor = TopKPathExtractor(graph,
                                  p, num_action,
                                  load_path='trained/model_pp_dqn.pth')
    ret = {}
    num_episodes = int(1e4)
    for i, (start, goal) in enumerate(scenarios):
        if i > 0: continue

        path_dict, sorted_items = extractor.rollout_paths(start, goal, num_episodes=num_episodes)
        print(i, len(scenarios))

        info = {}
        for k in np.linspace(0.1, 1.0, num=10):
            num = int(len(sorted_items) * k)
            paths = [path_dict[item[0]] for item in sorted_items[:num]]
            subgraph, nodes, edges = extractor.extract_subgraph(paths)

            ratio = len(nodes) / len(graph.nodes)
            print('\t>>>', len(sorted_items), end=' | ')
            print(round(k, 2), num, len(paths), end=' | ')
            print(round(ratio, 4))

            info[str(k)] = {'num_nodes': len(nodes),
                            'ratio': ratio,
                            'edges': edges}
            # plot_graph(graph, name='policy_attetion', subgraph=subgraph)
        ret['{}-{}'.format(start, goal)] = info

    with open('data/pa_{}.yaml'.format(num_episodes), 'w') as f:
        yaml.dump(ret, f)


# 运行主函数
if __name__ == "__main__":
    main()
