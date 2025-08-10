import numpy as np
import osmnx as ox
import networkx as nx

from common.utils import haversine
from env.rendering import StaticRender

base_speed = 45000.0 / 3600.0  # m/s


def extract_subgraph(graph, center, radius, render=False):
    center_node = list(graph.nodes)[center]
    nodes = graph.nodes

    sub_nodes = []
    for i, node in enumerate(list(nodes)):
        dist = haversine(nodes[center_node], nodes[node], km=True)
        if dist <= radius:
            sub_nodes.append(node)
    if render:
        ranges = {'center': center_node, 'radius': radius}
        StaticRender(graph).draw(vf=sub_nodes,
                                 ranges=ranges,
                                 name='fig_ranges_{}'.format(radius))
    subgraph = graph.subgraph(sub_nodes).copy()
    return subgraph, center_node


def load_graph(place_name,
               network_type='all',
               remove=True,
               center=None,
               radius=10.0,
               render=False,
               dynamic=False,
               threshold=None):
    print('Load Graph: {}'.format(place_name))
    graph = ox.graph_from_place(place_name, network_type=network_type)
    print(f"\t Initial {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print(f"\t Strong Connection: {nx.is_strongly_connected(graph)}")

    if center is not None:
        graph, center = extract_subgraph(graph, center, radius, render=render)
        print(f"\t After clipped by distance:", end=' ')
        print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")

    if threshold is not None:
        merge_near_nodes(graph, threshold=threshold)
        print(f"\t After merging near nodes:", end=' ')
        print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")

    if remove:
        g_strong = max(nx.strongly_connected_components(graph), key=len)
        graph = graph.subgraph(g_strong).copy()
        print(f"\t After strongly connected:", end=' ')
        print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")

    p, num_action = remove_isolated_nodes(graph)
    print(f"\t After removing isolated nodes:", end=' ')
    print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")

    info = add_dynamic_weight(graph, dynamic=dynamic)
    print(f"\t Strong Connection: {nx.is_strongly_connected(graph)}")
    print(f"\t Max cost: {info[-2]}, Min cost: {info[-1]}")
    print(f"\t Center node: ", center, center in list(graph.nodes))
    return graph, p, num_action, center


def remove_isolated_nodes(graph):
    """从图中移除孤立节点：没有连接任何其他节点的节点"""
    while True:
        isolated_nodes = []
        ret, max_number = {}, 0
        for node1 in graph.nodes:
            tmp = []
            for node2, data in graph[node1].items():
                if node1 == node2: continue
                if data.get('length', 1) > 0: tmp.append(node2)

            if len(tmp) <= 0:
                isolated_nodes.append(node1)
            else:
                ret[node1] = tmp
                max_number = max(max_number, len(tmp))

        if len(isolated_nodes) <= 0: break
        graph.remove_nodes_from(isolated_nodes)
    return ret, max_number


def merge_near_nodes(graph, threshold):
    """将低于阈值的相邻点合并"""
    while True:
        to_merge = []
        visited = set()
        for u, v, data in graph.edges(data=True):
            if u in visited or v in visited: continue

            dist = data.get('length', 1)
            if dist <= threshold:
                print(u, v, dist)
                to_merge.append((u, v, dist))
                visited.add(v)  # 避免重复合并

        print(f"\t\t Merged {len(to_merge)} pairs of nodes.")
        if len(to_merge) <= 0: break

        for u, v, d in to_merge:
            old_graph = graph.copy()
            for _, nbr, edge_data in old_graph.out_edges(v, data=True):
                if nbr == u: continue
                graph.add_edge(u, nbr, **edge_data)
            for nbr, _, edge_data in old_graph.in_edges(v, data=True):
                if nbr == u: continue
                graph.add_edge(nbr, u, **edge_data)
            if graph.has_node(v): graph.remove_node(v)


def add_dynamic_weight(graph, nodes=None, dynamic=False):
    if nodes is None: nodes = list(graph.nodes)

    max_cost, min_cost = 0.0, 1e6
    edge_index, edge_attr = [], []
    for u, v, data in graph.edges(data=True):
        times = 1.0
        if dynamic:
            times = np.random.uniform(0.1, 1.5)

        length = data.get('length', 1)
        traffic_time = length / (base_speed * times)
        data['times'] = times
        data['dynamic_weight'] = traffic_time
        if traffic_time > max_cost: max_cost = traffic_time
        if traffic_time < min_cost: min_cost = traffic_time

        i1 = nodes.index(u)
        i2 = nodes.index(v)
        edge_index.append([i1, i2])
        edge_attr.append(times)
        # edge_attr.append(traffic_time)
    return edge_index, edge_attr, max_cost, min_cost
