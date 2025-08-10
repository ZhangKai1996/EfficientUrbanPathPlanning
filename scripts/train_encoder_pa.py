import os
import shutil

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader

from algo.model import GraphEncoder
from common.geo import load_graph

from generate_scenarios import parse_config, parse_dynamics
from scripts.policy_attention import load_pa_table

_, scenarios = parse_config()
dynamics = parse_dynamics()
num_graphs = 100
suffix = 'encoder_pa'

num_paths = int(1e2)
k = int(1e2)
atte = load_pa_table(num_paths)


def nt_ent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # shape: (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # shape: (2B, 2B)

    # Remove self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, float('-inf'))
    # Positive similarities (i-th with i+batch_size)
    pos_sim = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    # Denominator: sum over all similarities (excluding self)
    sim_exp = torch.exp(sim)
    denom = sim_exp.sum(dim=1)
    # Numerator: exp(pos_sim)
    numerator = torch.exp(pos_sim)
    # Loss
    loss = -torch.log(numerator / (denom + 1e-8)).mean()
    return loss


def generate_graphs(graph):
    nodes = list(graph.nodes)
    state = np.zeros((len(nodes), ))
    for j, u in enumerate(nodes):
        state[j] = nodes.index(u)
    state0_tensor = torch.tensor(state, dtype=torch.int32)

    edge_index0, edge_attr0 = add_dynamic_weight(graph, dynamic=True)
    edge_index0_tensor = torch.tensor(edge_index0, dtype=torch.int32).T
    edge_attr0_tensor = torch.tensor(edge_attr0, dtype=torch.float32)
    data0 = Data(x=state0_tensor,
                 edge_index=edge_index0_tensor,
                 edge_attr=edge_attr0_tensor)

    datas = []
    for i in range(len(scenarios)):
        (start, end) = scenarios[i]
        node_list = atte['{}-{}'.format(start, end)]
        node_set = set([int(node) for node in node_list])
        if len(node_set) <= 0: continue
        subgraph = graph.subgraph(node_set).copy()

        sub_nodes = list(subgraph.nodes)
        state = np.zeros((len(sub_nodes), ))
        for j, u in enumerate(sub_nodes):
            state[j] = nodes.index(u)
        state_tensor = torch.tensor(state, dtype=torch.int32)

        edge_index, edge_attr = [], []
        for u, v, data in subgraph.edges(data=True):
            i1 = sub_nodes.index(u)
            i2 = sub_nodes.index(v)
            edge_index.append([i1, i2])
            edge_attr.append(data['dynamic_weight'])
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.int32).T
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        data = Data(x=state_tensor,
                    edge_index=edge_index_tensor,
                    edge_attr=edge_attr_tensor)
        datas.append(data)
    return data0, datas


def train(num_epochs=1000, log_folder='trained/logs_'+suffix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取某个地方的图（你可以根据需要改变位置）
    place_name = "南京航空航天大学(将军路校区)"
    graph, *_ = load_graph(place_name)

    model = GraphEncoder(input_dim=len(graph.nodes), output_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
        print('Removing all previous files!')
    writer = SummaryWriter(log_dir=log_folder)

    loss_stats = []
    for epoch in tqdm(range(num_epochs), desc='Encoder Training ...'):
        data0, dataset = generate_graphs(graph)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model.train()
        losses = []
        for data in train_loader:
            data = data.to(device)
            z1 = model(data.x, data.edge_index, data.edge_attr, data.batch)
            z2 = model(data0.x, data0.edge_index, data0.edge_attr, data0.batch)
            z2 = z2.repeat(z1.shape[0], 1)
            loss = nt_ent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_stats.append(np.mean(losses))

        if epoch % 10 == 0:
            mean_loss = np.mean(loss_stats)
            print('Epoch: {}, Loss: {:>7.2f}'.format(epoch, mean_loss))
            writer.add_scalar('loss', mean_loss, epoch)
            loss_stats = []
            torch.save(model.state_dict(),
                       'trained/model_{}_{}.pth'.format(suffix, epoch))


if __name__ == '__main__':
    train(num_epochs=int(1e4))
