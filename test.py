import numpy as np
from tqdm import tqdm

from baselines.pp_dijkstra import dijkstra_path
from common.geo import load_graph

from env import CityEnvironment


def run(env, render, reuse=False):
    env.reset(render=render, reuse=reuse)

    cost_di, path_di = dijkstra_path(env.graph,
                                     env.start_node,
                                     env.end_node,
                                     weight_key='dynamic_weight')
    print(path_di)
    cost_di = 0.0
    for i, u in enumerate(path_di[:-1]):
        v = path_di[i+1]
        cost_di += env.graph[u][v][0]['length'] / 1000.0
    return [cost_di, path_di]


def test_dqn(lst, num_episodes=int(1e3), render=False):
    labels = ['R_1', 'R_2', 'R_3']
    for ep in tqdm(range(num_episodes), desc='Test ...'):
        paths = {}
        env = None
        for i, env in enumerate(lst):
            result = run(env, render, reuse=True)
            key = '{}'.format(labels[i])
            paths[key] = [env.end_node, ] + result

        if render:
            env.render(paths=paths, name='fig_{}'.format(ep))


def main():
    np.random.seed(1234)

    c = 100.0
    ranges_v = [-1500.0, 0.0, 3.0]
    radius = 2e5
    env_kwargs = {
        'place_name': "Nanjing, China",
        'network_type': 'drive',
        'center': 0,
        'remove': True,
        'dynamic': True,
        'render': False
    }

    lst = []
    for i, v in enumerate(ranges_v):
        env_kwargs['dynamic'] = i in [1, ]
        graph, p, num_action, center_node = load_graph(radius=radius, **env_kwargs)
        env = CityEnvironment(graph, p, num_action, c=c, end_node=center_node)
        lst.append(env)

    test_dqn(lst, render=True, num_episodes=int(1e3))


if __name__ == '__main__':
    main()
