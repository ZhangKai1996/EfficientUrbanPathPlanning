import os

import numpy as np
import yaml

from common.geo import load_graph, add_dynamic_weight
from env.rendering import StaticRender

config_file = '../data/config.yaml'
dynamics_file = '../data/dynamics.yaml'


def plot_scenario():
    # 获取地图数据（可选择不同的网络类型，比如“drive”, “bike”, “walk”）
    network_type = 'drive'
    # place_name = "Jiangning, Nanjing, China"
    place_name = "Nanjing, China"
    graph, *_ = load_graph(place_name, network_type=network_type)

    paths, vo = [], []
    StaticRender(graph).draw(paths, vo)


class ScenarioGenerator:
    def __init__(self, place,
                 max_num_tasks=10,
                 max_len=100):
        self.speed = 45000.0 / 3600.0  # m/s
        self.graph, *_ = load_graph(place_name=place)

        self.max_num_tasks = max_num_tasks
        self.max_len = max_len
        self.others = None

    def generate_env_dynamics(self, num=1):
        nodes = list(self.graph.nodes)

        data = {}
        for i in range(num):
            edge_index, edge_attr = add_dynamic_weight(self.graph, dynamic=True)
            ret = {}
            for j, (i1, i2) in enumerate(edge_index):
                key = '{}-{}'.format(nodes[i1], nodes[i2])
                ret[key] = edge_attr[j]
            data[i] = ret

        with open(dynamics_file, 'w', encoding='utf-8') as f:
            yaml.dump(data=data, stream=f)
            print('Generate a new environment.')

    def generate_task_dynamics(self, num=1):
        data = {}
        for num_tasks in range(self.max_num_tasks, self.max_num_tasks+1):
            ret = {}
            for i in range(num):
                ret[i] = self.__generate_tasks(num_tasks)
            data[num_tasks] = ret

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(data=data, stream=f)
            print('Generate a new environment.')

    def __generate_tasks(self, num_tasks):
        all_nodes = list(range(len(self.graph.nodes)))
        # np.random.shuffle(all_nodes)
        ret = {'start': all_nodes[0], 'end': all_nodes[-1]}
        task_nodes = np.random.choice(all_nodes[1:-1], (num_tasks,), replace=False)
        task_nodes = [str(node) for node in task_nodes]
        ret['tasks'] = '-'.join(task_nodes)
        return ret


def generate_scenarios():
    # np.random.seed(1)
    kwargs = {
        "max_num_tasks": 3,
        "max_len": 100,
        "place": "南京航空航天大学(将军路校区)"
    }

    generator = ScenarioGenerator(**kwargs)
    generator.generate_task_dynamics(num=int(1e0))
    generator.generate_env_dynamics(num=int(1e1))


def parse_config():
    assert os.path.exists(config_file)
    with open(config_file, 'r') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)

    voo_scenarios = []
    opp_scenarios = []
    for num_tasks, ret in data.items():
        for i, info in ret.items():
            task_nodes = [int(node) for node in info['tasks'].split('-')]
            info['tasks'] = task_nodes
            voo_scenarios.append(info)

            nodes = [info['start'], info['end']] + task_nodes
            for node1 in nodes:
                for j, node2 in enumerate(nodes):
                    if j == 0 or j == len(nodes) - 1: continue
                    if node1 == node2: continue
                    opp_scenarios.append((node1, node2))
    opp_scenarios = list(set(opp_scenarios))
    print('VOO scenarios:', len(voo_scenarios))
    print('OPP scenarios:', len(opp_scenarios))
    return voo_scenarios, opp_scenarios


def parse_dynamics():
    assert os.path.exists(dynamics_file)
    with open(dynamics_file, 'r') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)

    dynamics = list(data.values())
    print('Dynamics instances: ', len(dynamics))
    return dynamics


# if __name__ == '__main__':
#     generate_scenarios()
#     parse_config()
#     parse_dynamics()
