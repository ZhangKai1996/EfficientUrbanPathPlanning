
import numpy as np
import matplotlib.pyplot as plt


markers = ['o', 's', 'D', 'v', '*', '+', 'x']
colors = ['r', 'g', 'b', 'orange', 'purple']


def plot_training_curves(ranges_c, ranges_v, root):
    print('---------------Training-----------------')
    labels = ['$\\eta$={}'.format(v*10) for v in ranges_v]
    labels[0] = '$R_2$'
    fig, axes = plt.subplots(3, 1)
    for c in ranges_c:
        for j, v in enumerate(ranges_v):
            rew_arr = load_csv(root + 'reward_{}_{}.csv'.format(c, v))[:, 2]
            result_arr = load_csv(root + 'result_{}_{}.csv'.format(c, v))
            result_arr = result_arr[:, [0, 3, 4, 5]]
            print(c, v, rew_arr.shape, result_arr.shape)

            rew_arr -= rew_arr.min()
            rew_arr /= abs(rew_arr.max())
            axes[0].plot(rew_arr, label=labels[j])
            axes[0].set_title('Training Curves', fontsize=18)
            axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axes[1].plot(result_arr[:, 1], label=labels[j])
            axes[1].set_title('Mean GAP', fontsize=18)
            axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axes[2].plot(result_arr[:, 3], label=labels[j])
            axes[2].set_title('Success Rate', fontsize=18)
            axes[2].set_xlabel('The number of Epoch', fontsize=16)
            axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    [ax.legend(fontsize=16) for ax in axes]
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
    ranges_c = np.linspace(30.0, 30.0, 1)
    ranges_v = [-1.5e3, 0.0, 3.0]
    root = 'trained/exp_dqn_{}/'.format(radius)
    print(root)

    plot_training_curves(ranges_c, ranges_v, root=root)


if __name__ == '__main__':
    main()
