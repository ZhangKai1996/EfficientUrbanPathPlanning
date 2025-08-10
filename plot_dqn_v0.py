
import numpy as np
import matplotlib.pyplot as plt


markers = ['o', 's', 'D', 'v', '*', '+', 'x']
colors = ['r', 'g', 'b', 'orange', 'purple']


def plot_training_curves(ranges_c, ranges_v, alpha, root):
    print('---------------Training-----------------')
    labels = ['$\\eta$={}'.format(v) for v in ranges_v]
    labels[0] = '$R_2$'

    fig, axes = plt.subplots(len(ranges_c), 2, constrained_layout=True)
    for i, c in enumerate(ranges_c):
        for j, v in enumerate(ranges_v):
            result_arr = load_csv(root + 'result_{}_{}_{}.csv'.format(c, v, alpha))
            result_arr = result_arr[:, [0, 3, 4, 5]]
            print(c, v, result_arr.shape)

            axes[i, 0].plot(result_arr[:, 1], label=labels[j])
            axes[i, 0].set_title('Mean GAP (c={})'.format(c), fontsize=18)
            axes[i, 0].set_ylim(bottom=0.0, top=0.4)
            # axes[i, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axes[i, 0].legend(fontsize=16)

            axes[i, 1].plot(result_arr[:, 3], label=labels[j])
            axes[i, 1].set_title('Success Rate (c={})'.format(c), fontsize=18)
            # axes[i, 1].set_xlabel('The number of Epoch', fontsize=16)
            # axes[i, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axes[i, 1].legend(fontsize=16)
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


def main():
    radius = int(3e4)
    ranges_c = np.linspace(6.0, 7.0, 2)
    ranges_v = [-1.5e3, 0.0, 1.0]
    alpha = 0.001
    root = 'trained/exp_dqn_{}/'.format(radius)
    print(root)

    plot_training_curves(ranges_c, ranges_v, alpha, root=root)


if __name__ == '__main__':
    main()
