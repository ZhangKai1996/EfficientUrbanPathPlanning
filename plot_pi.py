import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from train_pp_pi import evaluate_similarity

markers = ['o', 's', 'D', 'v', '*', '+', 'x']
colors = ['r', 'g', 'b', 'orange', 'purple']


def plot_v():
    seqs = [
        np.load('trained/v_d_2e4/v_rs_-1500.0(3000.0).npy'),
        np.load('trained/v_d_2e4/v_rs_-1000.0(3000.0).npy'),
        np.load('trained/v_d_2e4/v_rs_-500.0(3000.0).npy'),
        np.load('trained/v_d_2e4/v_rs_0.0(3000.0).npy'),
        np.load('trained/v_d_2e4/v_rs_500.0(3000.0).npy'),
        np.load('trained/v_d_2e4/v_rs_1000.0(3000.0).npy'),
        np.load('trained/v_d_2e4/v_rs_1500.0(3000.0).npy'),
        np.load('trained/v_d_2e4/v_rs_2000.0(3000.0).npy'),
        np.load('trained/v_d_2e4/v_rs_2500.0(3000.0).npy'),
    ]
    print(seqs[0].shape)

    labels = ['rs_-1.5', 'rs_-1', 'rs_-0.5', 'rs_0',
              'rs_1', 'rs_2', 'rs_3', 'rs_4', 'rs_5',
              'rs_6', 'rs_7', 'rs_8', 'rs_9', 'rs_10']

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
    for i, seq in enumerate(seqs):
        for j in range(4):
            axes[j].plot(seq[:, 215 + j] / max(seq[:, 215 + j]),
                         marker=markers[i % len(markers)],
                         markersize=3,
                         # color=colors[i % len(colors)],
                         label=labels[i])
            axes[j].legend()
            # axes[k].set_ylim(top=1000.0)
            # axes[k].set_xlim(0, 150)
    plt.show()


def plot_similarity(ranges_c, ranges_v, radius, root):
    print('---------------Similarity-----------------')
    num_c = len(ranges_c)
    num_v = len(ranges_v)

    x, y = [], []
    for i, c in enumerate(ranges_c):
        print(c)
        arr = np.load(root + 'v_d_{}_{}.npy'.format(c, radius))
        arr2 = optimality_stats(num_c, num_v, root)
        print(arr2.shape)

        x.append(c)
        metrics1 = evaluate_similarity(arr[:, 0], arr[:, 1])
        arr[:, 0] = (arr[:, 0] - np.mean(arr[:, 0])) / np.std(arr[:, 0])
        arr[:, 1] = (arr[:, 1] - np.mean(arr[:, 1])) / np.std(arr[:, 1])
        metrics2 = (evaluate_similarity(arr[:, 0], arr[:, 1]))
        print(metrics1)
        print(metrics2)
        print(arr2[i, 3])
        y.append(list(metrics1.values()) + list(metrics2.values()))

        # row, column = 3, 3
        # fig, axes = plt.subplots(row, column)
        # for i in range(row):
        #     for j in range(column):
        #         idx = i * 3 + j
        #         min_x, max_x = 100*idx, min(100*(idx+1), len(arr))
        #         x_list = list(range(min_x, max_x))
        #         axes[i, j].plot(x_list, arr[min_x:max_x, 0], label='$V^*(s;R_1)$')
        #         axes[i, j].plot(x_list, -arr[min_x:max_x, 1], label='$-R_2=d(s,s_g)$')
        #         axes[i, j].legend(loc='upper right')
        # plt.show()

    y = np.array(y)
    labels = ['pearson1', 'cosine1', 'mse1', 'slope1', 'intercept1',
              'pearson2', 'cosine2', 'mse2', 'slope2', 'intercept2', ]
    row, column = 2, 5
    fig, axes = plt.subplots(row, column)
    for i in range(row):
        for j in range(column):
            idx = i * 5 + j
            axes[i, j].plot(x, y[:, idx], label=labels[idx])
            axes[i, j].legend(loc='upper right')
    plt.show()


def load_csv(file_path):
    with open(file_path, 'r', newline='') as f:
        results = []
        for line in f.readlines()[:]:
            result = line.strip('\r\n').split(',')
            result = [float(x) for x in result]
            results.append(result)
        result_arr = np.array(results)
    return result_arr


def plot_c(ranges_c, ranges_v, root):
    print('--------------------c---------------------')
    arr = load_csv(root + 'result_dpp_pi.csv')

    arr = arr.reshape((len(ranges_c), len(ranges_v), -1))
    print(arr.shape)
    arr = arr[:, :, [1, 2, 6, 8]]
    print(arr.shape)
    arr0 = arr.transpose(1, 0, 2)
    print(arr0.shape)

    y_labels = ['Mean GAP', 'Success Rate']

    fig, axes = plt.subplots(1, 2, constrained_layout=False)
    for row in range(14):
        for column in range(2):
            x_list = arr0[row, :, 1] / 1e3
            y_list = arr0[row, :, 2 + column]
            label = '$\\eta$={}'.format(arr0[row, 0, 0] / 1e3)
            if arr0[row, 0, 0] < -1000.0:
                label = 'R_2'
            axes[column].plot(x_list, y_list,
                              marker=markers[row % len(markers)],
                              label=label)
            axes[column].grid(axis='y', linestyle='--', alpha=0.5)
            axes[column].set_xlabel('Task Intention Factor $c$', fontsize=18)
            axes[column].set_ylabel(y_labels[column], fontsize=18)
            axes[column].legend(fontsize=16)
    plt.show()


def plot_c_1(ranges_c, ranges_v, root):
    print('--------------------c---------------------')
    arr = load_csv(root + 'result_dpp_pi.csv')

    arr = arr.reshape((len(ranges_c), len(ranges_v), -1))
    print(arr.shape)
    arr = arr[:, :, [1, 2, 6, 8]]
    print(arr.shape)
    arr0 = arr.transpose(1, 0, 2)
    print(arr0.shape)

    titles = ['Mean GAP (Lower is better)', 'Success Rate (Higher is better)']
    cmaps = ['YlGnBu', 'magma']

    y_labels = ['{}'.format(x / 1e3) for x in ranges_v]
    y_labels[0] = 'R_2'
    x_labels = ['{}'.format(x / 1e3) for x in ranges_c]
    y_ticks = list(range(0, len(ranges_v)))
    x_ticks = list(range(0, len(ranges_c)))

    plt.figure()
    data = arr0[:, :, 2]
    im = plt.imshow(data, cmap=cmaps[0])
    # plt.colorbar(im)

    # 在格子中写数值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")

    plt.ylabel('Shaping coefficient $\\eta$', fontsize=18)
    plt.xlabel('Task Intention Factor $c$', fontsize=18)
    plt.title(titles[0], fontsize=18)
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45)
    plt.yticks(ticks=y_ticks, labels=y_labels, rotation=45)
    plt.show()

    plt.figure()
    data = arr0[:, :, 3]
    im = plt.imshow(data, cmap=cmaps[1])
    # plt.colorbar(im)

    # 在格子中写数值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")

    plt.ylabel('Shaping coefficient $\\eta$', fontsize=18)
    plt.xlabel('Task Intention Factor $c$', fontsize=18)
    plt.title(titles[1], fontsize=18)
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45)
    plt.yticks(ticks=y_ticks, labels=y_labels, rotation=45)
    plt.show()

    # fig, axes = plt.subplots(1, 2, constrained_layout=True)
    # for row in range(2):
    #     ax = axes[row]
    #     data = arr0[:, :, 2+row]
    #     im = ax.imshow(data, cmap=cmaps[row])
    #     plt.colorbar(im)
    #
    #     # 在格子中写数值
    #     for i in range(data.shape[0]):
    #         for j in range(data.shape[1]):
    #             ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")
    #
    #     ax.set_ylabel('Shaping coefficient $\\eta$', fontsize=18)
    #     ax.set_xlabel('Task Intention Factor $c$', fontsize=18)
    #     ax.set_title(titles[row], fontsize=18)
    #     ax.set_xticks(ticks=x_ticks, labels=x_labels, rotation=45)
    #     ax.set_yticks(ticks=y_ticks, labels=y_labels, rotation=45)
    # plt.show()


def optimality_stats(num_c, num_v, root):
    print('Load optimality from result_dpp_pi.csv')
    with open(root + 'result_dpp_pi.csv', 'r', newline='') as f:
        results = []
        for line in f.readlines():
            result = line.strip('\r\n').split(',')
            result = [float(x) for x in result]
            results.append(result)
        result_arr = np.array(results)
        print('\t', result_arr.shape)
        result_arr = result_arr.reshape((num_c, num_v, -1))
        print('\t', result_arr.shape)
        result_arr = result_arr[:, :, [6, 7, 8]]
        print('\t', result_arr.shape)
        return result_arr


def efficiency_stats(c, ranges_v, root):
    print('Compute efficiency from v_rs({}).npy'.format(c))
    seqs = [np.load(root + 'v_rs_{}({}).npy'.format(x, c)) for x in ranges_v]
    print('\t', seqs[0].shape)

    lst, max_iters = [], []
    for i, seq in enumerate(seqs):
        min_iters = []
        for s in range(seq.shape[-1]):
            v = seq[:, s]
            k = 0
            max_v = v.max() * 0.95
            for k, x in enumerate(v):
                if x >= max_v: break
            min_iters.append(k)
        lst.append(np.array(min_iters))
        max_iters.append(max(min_iters))

    arr = np.array(lst)
    print(arr, arr[3, :])
    print(arr - arr[3, :])
    arr = (arr[3, :] - arr) / arr
    print(arr)
    arr1 = np.array(max_iters)
    return arr, arr1


def plot_v_efficiency(ranges_c, ranges_v, root):
    print('------------v_efficiency----------------')
    num_c = len(ranges_c)
    num_v = len(ranges_v)
    arr2 = optimality_stats(num_c, num_v, root)

    labels = ['{}'.format(x / 1e3) for x in ranges_v]
    labels[0] = 'R_2'
    x_ticks = list(range(1, num_v + 1))
    kwargs = {'sharex': True, 'sharey': True, 'constrained_layout': True}

    fig, axes = plt.subplots(2, 6, **kwargs)
    lst = []
    for i, c in enumerate(ranges_c):
        print('{}. c={}'.format(i, c))
        row, column = i // 6, i % 6

        x_list = list(range(1, num_v + 1))
        arr, arr1 = efficiency_stats(c, ranges_v, root)
        lst.append([arr, arr1])
        if arr2[i, 3, 2] == 1.0:
            axes[row, column].boxplot(arr.transpose([1, 0]), showfliers=False)
            axes[row, column].plot(x_list, (arr1[3] - arr1) / arr1)

        axes[row, column].set_xticks(ticks=x_ticks, labels=labels, rotation=45)
        axes[row, column].grid(axis='y', linestyle='--', alpha=0.5)
        axes[row, column].set_ylabel('The ratio (c={})'.format(c))
    plt.show()

    fig, axes = plt.subplots(2, 6, **kwargs)
    for i, c in enumerate(ranges_c):
        print('{}. c={}'.format(i, c))
        row, column = i // 6, i % 6

        x_list = list(range(1, num_v + 1))
        arr, arr1 = lst[i]
        if arr2[i, 3, 2] == 1.0:
            y_list = list(arr.mean(axis=1))
            assert len(x_list) == len(y_list)
            axes[row, column].plot([1, ], y_list[0:1], marker='^', label='Improved Efficiency ($R_2$)')
            axes[row, column].plot(x_list[1:], y_list[1:], marker='^', label='Improved Efficiency')

        axes[row, column].set_xticks(ticks=x_ticks, labels=labels, rotation=45)
        axes[row, column].grid(axis='y', linestyle='--', alpha=0.5)
        axes[row, column].legend(loc='upper left')
        axes[row, column].set_ylim(bottom=-0.5, top=0.5)

        axes1_2 = axes[row, column].twinx()
        axes1_2.plot([1, ], [arr2[i, 0, 0], ], marker='*', label='Mean GAP($R_2$)')
        axes1_2.plot(x_list[1:], arr2[i, 1:, 0], marker='*', label='Mean GAP')
        axes1_2.plot([1, ], [arr2[i, 0, 2], ], marker='+', label='Success Rate($R_2$)')
        axes1_2.plot(x_list[1:], arr2[i, 1:, 2], marker='+', label='Success Rate')
        axes1_2.set_ylim(bottom=0.0, top=1.5)
        axes1_2.legend(loc='upper right')
    plt.show()

    fig, axes = plt.subplots(1, 2)
    for i, c in enumerate(ranges_c):
        x_list = list(range(1, num_v + 1))
        arr, arr1 = lst[i]

        ids = np.argwhere(arr2[i, 1:, 2] >= 1.0)[:, 0]
        axes[0].plot([x_list[m] for m in ids], arr1[ids], label='c={}'.format(c))
        axes[0].set_xticks(ticks=x_ticks, labels=labels, rotation=45)
        axes[0].legend()

        axes[1].plot(x_list, arr2[i, :, 2], label='c={}'.format(c))
        axes[1].set_xticks(ticks=x_ticks, labels=labels, rotation=45)
        axes[1].legend()

    plt.show()


def plot_v_efficiency_1(ranges_c, ranges_v, root):
    print('------------v_efficiency----------------')
    num_c = len(ranges_c)
    num_v = len(ranges_v)
    arr2 = optimality_stats(num_c, num_v, root)

    labels = ['{}'.format(x / 1e3) for x in ranges_v]
    labels[0] = 'R_2'
    x_ticks = list(range(1, num_v + 1))
    kwargs = {'sharex': True, 'sharey': False, 'constrained_layout': False}

    fig, axes = plt.subplots(1, 2, **kwargs)
    cs = [1e3, ]

    for i, c in enumerate(cs):
        print('{}. c={}'.format(i, c))

        x_list = list(range(1, num_v + 1))
        arr, arr1 = efficiency_stats(c, ranges_v, root)
        axes[0].boxplot(arr.transpose([1, 0]), showfliers=False)
        axes[0].plot(x_list, (arr1[3] - arr1) / arr1)
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Shaping Coefficient $\\eta$', fontsize=18)
        axes[0].set_title('c={}'.format(c/1e3), fontsize=18)
        axes[0].set_xticks(ticks=x_ticks, labels=labels, rotation=45)

        if i == 0:
            axes[0].set_ylabel('Improved Efficiency $h_\\eta$', fontsize=18)

        x_list = list(range(1, num_v + 1))
        arr, arr1 = efficiency_stats(c, ranges_v, root)
        y_list = list(arr.mean(axis=1))
        assert len(x_list) == len(y_list)
        axes[1].plot([1, ], y_list[0:1], marker='^', label='Improved Efficiency($R_2$)')
        axes[1].plot(x_list[1:], y_list[1:], marker='^', label='Improved Efficiency($\\eta$)')

        axes[1].set_xticks(ticks=x_ticks, labels=labels, rotation=45)
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)
        axes[1].set_title('c={}'.format(c/1e3), fontsize=18)
        axes[1].set_xlabel('Shaping Coefficient $\\eta$', fontsize=18)
        axes[1].set_ylim(bottom=-0.1, top=0.5)
        axes[1].legend(loc='upper left', fontsize=16)

        axes1_2 = axes[1].twinx()
        idx = list(ranges_c).index(c)
        axes1_2.plot([1, ], [arr2[idx, 0, 0], ], marker='*', label='Mean GAP($R_2$)')
        axes1_2.plot(x_list[1:], arr2[idx, 1:, 0], marker='*', label='Mean GAP($\\eta$)')
        axes1_2.plot([1, ], [arr2[idx, 0, 2], ], marker='+', label='Success Rate($R_2$)')
        axes1_2.plot(x_list[1:], arr2[idx, 1:, 2], marker='+', label='Success Rate($\\eta$)')
        axes1_2.set_ylim(bottom=-0.1, top=1.5)
        axes1_2.legend(loc='upper right', fontsize=16)

        if i == 0:
            axes[1].set_ylabel('Mean Improved Efficiency', fontsize=18)
        if i == len(cs) - 1:
            axes1_2.set_ylabel('GAP or Success Rate', fontsize=18)

    plt.show()


def plot_v_efficiency_2(ranges_c, ranges_v, root):
    print('------------v_efficiency----------------')
    num_c = len(ranges_c)
    num_v = len(ranges_v)
    arr2 = optimality_stats(num_c, num_v, root)

    labels = ['{}'.format(x / 1e3) for x in ranges_v]
    labels[0] = 'R_2'
    x_ticks = list(range(1, num_v + 1))
    kwargs = {'sharex': True, 'sharey': True, 'constrained_layout': True}

    cs = [1e3, 1.5e3, 2e3, 2.5e3, 3e3]
    fig, axes = plt.subplots(1, len(cs), **kwargs)

    for i, c in enumerate(cs):
        print('{}. c={}'.format(i, c))

        x_list = list(range(1, num_v + 1))
        arr, arr1 = efficiency_stats(c, ranges_v, root)
        axes[i].boxplot(arr.transpose([1, 0]), showfliers=False)
        axes[i].plot(x_list, (arr1[3] - arr1) / arr1)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        axes[i].set_title('c={}'.format(c/1e3), fontsize=18)
        axes[i].set_xticks(ticks=x_ticks, labels=labels, rotation=45)

        if i == 0:
            axes[i].set_ylabel('Improved Efficiency $\\delta_\\eta$', fontsize=18)

    plt.show()

    fig, axes = plt.subplots(1, 5, **kwargs)
    for i, c in enumerate(cs):
        print('{}. c={}'.format(i, c))

        x_list = list(range(1, num_v + 1))
        arr, arr1 = efficiency_stats(c, ranges_v, root)
        y_list = list(arr.mean(axis=1))
        assert len(x_list) == len(y_list)
        axes[i].plot([1, ], y_list[0:1], marker='^', label='$\\delta$($R_2$)')
        axes[i].plot(x_list[1:], y_list[1:], marker='^', label='$\\delta_\\eta$)')

        axes[i].set_xticks(ticks=x_ticks, labels=labels, rotation=45)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        axes[i].set_title('c={}'.format(c/1e3), fontsize=18)
        # axes[i].set_xlabel('$\\eta$', fontsize=18)
        axes[i].set_ylim(bottom=-0.1, top=0.5)
        axes[i].legend(loc='upper left')

        axes1_2 = axes[i].twinx()
        idx = list(ranges_c).index(c)
        axes1_2.plot([1, ], [arr2[idx, 0, 0], ], marker='*', label='Mean GAP($R_2$)')
        axes1_2.plot(x_list[1:], arr2[idx, 1:, 0], marker='*', label='Mean GAP($\\eta$)')
        axes1_2.plot([1, ], [arr2[idx, 0, 2], ], marker='+', label='Success Rate($R_2$)')
        axes1_2.plot(x_list[1:], arr2[idx, 1:, 2], marker='+', label='Success Rate($\\eta$)')
        axes1_2.set_ylim(bottom=-0.1, top=1.5)
        axes1_2.legend(loc='upper right')

        if i == 0:
            axes[i].set_ylabel('Mean Improved Efficiency', fontsize=18)

        if i == len(cs) - 1:
            axes1_2.set_ylabel('GAP or Success Rate', fontsize=18)
        else:
            axes1_2.set_yticks([])

    plt.show()


def main():
    radius = int(3e4)
    ranges_c = np.linspace(0.0, 5e3, 11)
    ranges_v = np.linspace(-1500.0, 5e3, 14)
    root = 'trained/exp_{}/'.format(radius)
    print(root)

    # plot_similarity(ranges_c, ranges_v, radius=radius, root=root)
    # plot_c_1(ranges_c, ranges_v, root=root)
    # plot_v_efficiency_1(ranges_c, ranges_v, root=root)
    plot_v_efficiency_2(ranges_c, ranges_v, root=root)


if __name__ == '__main__':
    main()
