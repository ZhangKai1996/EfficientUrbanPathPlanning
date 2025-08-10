import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from algo.memory import SimpleReplayBuffer
from common.geo import load_graph, add_dynamics

from generate_scenarios import parse_config, parse_dynamics

_, scenarios = parse_config()
dynamics = parse_dynamics()
num_graphs = 100
suffix = 'encoder'


class DQNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQNEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def nt_ent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # 去掉自相似项
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, float('-inf'))

    # 正样本对：z1[i] 与 z2[i]
    pos_sim = torch.cat([
        torch.diag(sim, batch_size),
        torch.diag(sim, -batch_size)
    ], dim=0)  # (2B,)

    # 使用 logsumexp 替代 log(sum(exp(.)))，更稳定
    denom = torch.logsumexp(sim, dim=1)  # (2B,)

    # 正样本在 sim 矩阵中的 logit 对应值就是 pos_sim
    loss = - (pos_sim - denom).mean()
    return loss


class GraphEncoder:
    def __init__(self, graph,
                 buffer_size=int(1e6),
                 has_writer=True,
                 log_folder='trained/logs_'+suffix):
        self.graph = graph
        self.model = DQNEncoder(input_dim=len(graph.edges), output_dim=32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.replay_buffer = SimpleReplayBuffer(capacity=buffer_size)

        self.writer = None
        if has_writer:
            if os.path.exists(log_folder):
                shutil.rmtree(log_folder)
                print('Removing all previous files!')
            self.writer = SummaryWriter(log_dir=log_folder)

    def generate_graphs(self):
        for i in range(num_graphs):
            _, edge_attr_1 = add_dynamics(self.graph, dynamics[i])
            for j in range(num_graphs):
                _, edge_attr_2 = add_dynamics(self.graph, dynamics[j])
                self.replay_buffer.push(edge_attr_1, edge_attr_2)

    def test(self):
        self.generate_graphs()
        batch_size = len(self.replay_buffer)
        batch_size = 64

        plt.figure(figsize=(30, 20))
        batch = self.replay_buffer.sample(batch_size=batch_size)
        x_1, x_2 = zip(*batch)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)

        x_1_tensor = torch.tensor(np.array(x_1), dtype=torch.float32)
        x_2_tensor = torch.tensor(np.array(x_2), dtype=torch.float32)
        embeddings0 = x_1_tensor.detach().numpy()
        emb2_2d = tsne.fit_transform(embeddings0)
        axes[0, 0].scatter(emb2_2d[:, 0], emb2_2d[:, 1], s=20, alpha=0.5, label=f'x1')
        axes[0, 0].set_title('t-SNE (0)')
        axes[0, 0].legend()

        state_dict = torch.load('trained/model_{}_0.pth'.format(suffix))
        self.model.load_state_dict(state_dict)
        embeddings1 = self.model(x_1_tensor)
        embeddings = embeddings1.detach().numpy()
        emb1_2d = tsne.fit_transform(embeddings)
        axes[0, 1].scatter(emb1_2d[:, 0], emb1_2d[:, 1], s=20, alpha=0.5, label=f'x1')
        axes[0, 1].set_title('t-SNE (1)')
        axes[0, 1].legend()

        state_dict = torch.load('trained/model_{}_10.pth'.format(suffix))
        self.model.load_state_dict(state_dict)
        embeddings1 = self.model(x_1_tensor)
        embeddings = embeddings1.detach().numpy()
        emb1_2d = tsne.fit_transform(embeddings)
        axes[0, 2].scatter(emb1_2d[:, 0], emb1_2d[:, 1], s=20, alpha=0.5, label=f'x1')
        axes[0, 2].set_title('t-SNE (2)')
        axes[0, 2].legend()

        state_dict = torch.load('trained/model_{}_20.pth'.format(suffix))
        self.model.load_state_dict(state_dict)
        embeddings1 = self.model(x_1_tensor)
        embeddings = embeddings1.detach().numpy()
        emb1_2d = tsne.fit_transform(embeddings)
        axes[1, 0].scatter(emb1_2d[:, 0], emb1_2d[:, 1], s=20, alpha=0.5, label=f'x1')
        axes[1, 0].set_title('t-SNE (3)')
        axes[1, 0].legend()

        state_dict = torch.load('trained/model_{}_30.pth'.format(suffix))
        self.model.load_state_dict(state_dict)
        embeddings1 = self.model(x_1_tensor)
        embeddings = embeddings1.detach().numpy()
        emb1_2d = tsne.fit_transform(embeddings)
        axes[1, 1].scatter(emb1_2d[:, 0], emb1_2d[:, 1], s=20, alpha=0.5, label=f'x1')
        axes[1, 1].set_title('t-SNE (4)')
        axes[1, 1].legend()

        state_dict = torch.load('trained/model_{}_40.pth'.format(suffix))
        self.model.load_state_dict(state_dict)
        embeddings1 = self.model(x_1_tensor)
        embeddings = embeddings1.detach().numpy()
        emb1_2d = tsne.fit_transform(embeddings)
        axes[1, 2].scatter(emb1_2d[:, 0], emb1_2d[:, 1], s=20, alpha=0.5, label=f'x1')
        axes[1, 2].set_title('t-SNE (5)')
        axes[1, 2].legend()

        plt.savefig('trained/t-SNE.png', dpi=300)

    def train(self, num_epochs=1000):
        self.generate_graphs()
        batch_size = len(self.replay_buffer)
        print(batch_size)

        losses = []
        for epoch in range(num_epochs):
            self.model.train()
            batch = self.replay_buffer.sample(batch_size=batch_size)
            x_0, x_1 = zip(*batch)
            x_0_tensor = torch.tensor(np.array(x_0), dtype=torch.float32)
            x_1_tensor = torch.tensor(np.array(x_1), dtype=torch.float32)

            z1 = self.model(x_0_tensor)
            z2 = self.model(x_1_tensor)
            loss = nt_ent_loss(z1, z2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if len(losses) <= 0: continue
            if epoch % 10 == 0:
                mean_loss = np.mean(losses)
                print('Epoch: {}/{}, Loss: {:>7.4f}'.format(epoch, num_epochs, mean_loss))
                self.writer.add_scalar('loss', mean_loss, epoch)
                torch.save(self.model.state_dict(),
                           'trained/model_{}_{}.pth'.format(suffix, epoch))
                losses = []


def main():
    place_name = "南京航空航天大学(将军路校区)"
    graph, *_ = load_graph(place_name)

    encoder = GraphEncoder(graph)
    # encoder.train(num_epochs=int(5e1))
    encoder.test()


if __name__ == '__main__':
    main()
