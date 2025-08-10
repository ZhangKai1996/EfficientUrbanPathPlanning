import csv
import os.path
import time
import shutil

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from baselines.pp_dijkstra import dijkstra_path
from common.geo import load_graph
from common.utils import haversine, softmax

from env import CityEnvironment
from algo import PPDQNTrainer

train_max_step = 100
test_max_step = 100


def train_dqn(graph, p,
              num_action,
              center_node,
              eta=0.0,
              c=0.0,
              alpha=1.0,
              save_rate=None,
              batch_size=32,
              num_episodes=int(1e3),
              epsilon_decay=0.1,
              epsilon_end=0.1,
              plot_reward=False,
              folder='exp_dqn',
              suffix='0_0'):
    env = CityEnvironment(graph, p, num_action, c=c, eta=eta, alpha=alpha, end_node=center_node)
    trainer = PPDQNTrainer(graph, p, num_action,
                           batch_size=batch_size,
                           epsilon_decay=int(num_episodes * epsilon_decay),
                           epsilon_end=epsilon_end)

    if save_rate is None: save_rate = len(graph.nodes)

    root = 'trained/{}/'.format(folder)
    if plot_reward:
        path_distribution(env, num_episodes=int(1e4), name=root+'fig_' + suffix)

    log_folder = root + '/logs_{}'.format(suffix)
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
        print('Removing all previous files!')
    writer = SummaryWriter(log_dir=log_folder)

    rew_stats, sr_stats = [], []
    loss_stats, step_stats = [], []
    step, start_time = 0, time.time()
    for episode in tqdm(range(1, num_episodes + 1), desc='Training ...'):
        done, (state, mask) = False, env.reset()

        total_reward, episode_step = 0.0, 0
        while not done:
            action = trainer.choose_action(state, mask, episode=episode)
            (next_state, next_mask), reward, done, *_ = env.step(action)
            experience = (state, mask, action, reward, next_state, next_mask, float(done))
            trainer.replay_buffer.push(*experience, key=env.end_node_idx)
            loss = trainer.update_q_network(t=step)
            if loss is not None:
                loss_stats.append(loss)

            state, mask = next_state, next_mask
            total_reward += reward
            episode_step += 1
            step += 1
            if episode_step >= train_max_step: break

        step_stats.append(episode_step)
        rew_stats.append(total_reward)
        sr_stats.append(int(done))
        if episode % save_rate == 0:
            end_time = time.time()
            mean_rew = np.mean(rew_stats)
            mean_step = np.mean(step_stats)
            sr = np.mean(sr_stats)
            eps = trainer.epsilon(episode)

            print("Episode: {}".format(episode), end=', ')
            print("Step: {}".format(step), end=', ')
            print("Mean Step: {:>6.2f}".format(mean_step), end=', ')
            print("Mean Rew: {:>7.2f}".format(mean_rew), end=', ')
            print("SR: {:>6.4f}".format(sr), end=', ')
            print("Epsilon: {:>6.4f}".format(eps), end=',')
            print("Time: {:>5.2f}".format(end_time - start_time))

            if len(loss_stats) > 0:
                mean_loss = np.mean(loss_stats)
                writer.add_scalar('Loss', mean_loss, episode)
            writer.add_scalar('Step', mean_step, episode)
            writer.add_scalar('Reward', mean_rew, episode)
            writer.add_scalar('Success Rate', sr, episode)
            writer.add_scalar('Epsilon', eps, step)

            trainer.save_model(root+'model_{}.pth'.format(suffix))
            rew_stats, sr_stats = [], []
            loss_stats, step_stats = [], []
            start_time = end_time

            test_dqn(env, trainer,
                     epoch=episode // save_rate,
                     num_episodes=save_rate,
                     test_path=root + 'result_{}'.format(suffix))


def test_dqn(env, trainer,
             num_episodes=int(1e3),
             epoch=0,
             load_path=None,
             render=False,
             test_path=None):
    if load_path is not None:
        trainer.load_mode(load_path)

    sr_stats, gap_stats = [], []
    for ep in tqdm(range(num_episodes), desc='Testing ...'):
        done_rl, (state, mask) = False, env.reset(render=render)

        cost_rl, episode_step = 0.0, 0
        path_rl = [env.cur_node, ]
        while not done_rl:
            action = trainer.choose_action(state, mask)
            (next_state, next_mask), reward, done_rl = env.step(action, is_test=True)
            path_rl.append(env.cur_node)
            cost_rl += -reward
            state, mask = next_state, next_mask
            episode_step += 1
            if episode_step >= test_max_step: break

        start_node, end_node = env.start_node, env.end_node
        cost_di, path_di = dijkstra_path(env.graph, start_node, end_node, weight_key='length')
        cost_di *= env.alpha
        done_di = path_di[0] == start_node and path_di[-1] == end_node

        if render:
            env.render([path_rl, ], vo=[start_node, end_node])

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
        result = [epoch, min_gap[0], max_gap[0]] + list(mean_gap) + list(mean_sr)
    else:
        result = [epoch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if test_path is not None:
        with open(test_path+'.csv', 'a+', newline='') as f:
            csv.writer(f).writerow(result)


def choose_action(env, beta=1.0, random=False):
    next_nodes = env.p[env.cur_node]
    nodes = env.graph.nodes
    actions = list(range(len(next_nodes)))
    if random:
        return np.random.choice(len(actions))

    dists = [-haversine(nodes[next_node], nodes[env.end_node])*beta for next_node in next_nodes]
    dist_array = np.array(dists)
    dist_array -= dist_array.min()
    probs = softmax(dist_array)
    return np.random.choice(actions, p=probs)


def path_distribution(env, name='fig', num_episodes=int(1e4)):
    path_cost = {'0': [], '1': []}
    check_dict = {}
    for i in range(num_episodes):
        done, (state, mask) = False, env.reset()

        total_reward, episode_step = 0.0, 0
        while not done:
            action = choose_action(env, random=True)
            (next_state, next_mask), reward, done, *_ = env.step(action)

            if next_state[1] in check_dict.keys():
                check_dict[next_state[1]] += 1
            else:
                check_dict[next_state[1]] = 0

            total_reward += reward
            episode_step += 1
            if episode_step >= train_max_step:
                break

        key = '{}'.format(int(done))
        path_cost[key].append([i, total_reward, episode_step])
    env.render(vf=check_dict)

    fig, axes = plt.subplots(2, 1, constrained_layout=True)
    count, title = 0, ''

    record_array_0 = np.array(path_cost['0'])
    if len(record_array_0) > 0:
        length = len(record_array_0[:, 1])
        xs = np.array(list(range(length)))
        ys = np.sort(record_array_0[:, 1])
        count += length

        axes[0].scatter(xs, ys, color='blue', label='done', alpha=0.6)
        title += '0: {}, '.format(length)

    record_array_1 = np.array(path_cost['1'])
    if len(record_array_1) > 0:
        length = len(record_array_1[:, 1])
        xs = np.array(list(range(count, count + length)))
        ys = np.sort(record_array_1[:, 1])
        count += length
        axes[0].scatter(xs, ys, color='red', label='not done', alpha=0.6)
        title += '1: {}, '.format(length)

    [ax.legend() for ax in axes]
    axes[0].set_title(title)
    plt.savefig(name + '.png', dpi=300)
    # plt.show()
    plt.close()


def main():
    np.random.seed(1234)

    c = 30.0
    eta = -1500.0
    alpha = 0.001
    radius = 7e4
    env_kwargs = {
        'place_name': "Nanjing, China",
        'network_type': 'drive',
        'center': 0,
        'remove': True,
        'render': False
    }
    kwargs = {
        'num_episodes': int(2e6),
        'epsilon_decay': 0.5,
        'epsilon_end': 0.1,
        'batch_size': 256,
        'plot_reward': False,
        'folder': 'exp_dqn_{}'.format(int(radius)),
        'suffix': '{}_{}_{}'.format(c, eta, alpha)
    }

    args = load_graph(radius=radius, **env_kwargs)
    train_dqn(*args, eta=eta, c=c, alpha=alpha, **kwargs)


if __name__ == '__main__':
    main()
