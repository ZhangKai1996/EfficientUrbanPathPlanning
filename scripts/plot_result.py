import numpy as np
import matplotlib.pyplot as plt


def main(name='fig'):
    with open('result_{}.csv'.format(suffix), 'w', newline='') as f:
        for line in f.readlines():
            continue

    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    mean_values = np.array(stats).mean(axis=0)
    axes[0].hist(gap_stats, bins=20, alpha=0.6, density=True)
    title = 'SR(rl):{:>6.3f}, '.format(mean_values[0])
    title += 'SR(di):{:>6.3f}'.format(mean_values[2])
    axes[0].set_title(title)

    axes[1].boxplot(gap_stats, notch=True, showmeans=True)
    title = 'GAP:{:>6.3f}'.format(np.mean(gap_stats))
    axes[1].set_title(title)

    plt.savefig(name + '.png', dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    main()
