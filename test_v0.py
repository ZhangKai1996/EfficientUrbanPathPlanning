import numpy as np
from tqdm import tqdm

from baselines.pp_dijkstra import dijkstra_path
from common.geo import load_graph

from env import CityEnvironment
from algo import PPDQNTrainer

max_step = 100


def run(env, trainer, render, reuse=False):
    state, mask = env.reset(render=render, reuse=reuse)
    done_rl = False

    cost_rl, episode_step = 0.0, 0
    path_rl = [env.cur_node, ]
    while not done_rl:
        action = trainer.choose_action(state, mask)
        (next_state, next_mask), reward, done_rl = env.step(action, is_test=True)
        path_rl.append(env.cur_node)
        cost_rl += -reward
        state, mask = next_state, next_mask
        episode_step += 1
        if episode_step >= max_step: break
    return [cost_rl, path_rl, done_rl]


def test_dqn(env, settings, num_episodes=int(1e3), render=False):
    for ep in tqdm(range(num_episodes), desc='Test ...'):
        paths = {}
        for i, (key, trainer) in enumerate(settings.items()):
            result = run(env, trainer, render, reuse=i >= 1)
            paths[key] = [env.end_node, ] + result

        start_node, end_node = env.start_node, env.end_node
        cost_di, path_di = dijkstra_path(env.graph, start_node, end_node, weight_key='length')
        cost_di /= 1000.0
        done_di = path_di[0] == start_node and path_di[-1] == end_node
        paths['di'] = [end_node, cost_di, path_di, done_di]

        if render:
            env.render(paths=paths, name='fig_{}'.format(ep))


def main():
    np.random.seed(1234)

    c = 30.0
    ranges_v = [-1500.0, 0.0, 1.0, 2.0, 3.0]
    radius = 3e4
    env_kwargs = {
        'place_name': "Nanjing, China",
        'network_type': 'drive',
        'center': 0,
        'remove': True,
        'render': False
    }

    graph, p, num_action, center_node = load_graph(radius=radius, **env_kwargs)
    env = CityEnvironment(graph, p, num_action, c=c, end_node=center_node)
    root = 'trained/exp_dqn_{}/'.format(int(radius))

    ret = {}
    for v in ranges_v:
        suffix = '{}_{}'.format(c, v)
        trainer = PPDQNTrainer(graph, p, num_action)
        trainer.load_mode(root+'model_{}.pth'.format(suffix))
        ret[suffix] = trainer

    test_dqn(env, ret, render=True, num_episodes=len(graph.nodes))


if __name__ == '__main__':
    main()
