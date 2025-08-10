import time

from common.geo import load_graph, base_speed
from env.rendering import StaticRender

from generate_scenarios import parse_config, parse_dynamics
from algo.pp_solver import *
from algo.tsp_solver import *

max_step = 100
voo_scenarios, *_ = parse_config()
dynamics = parse_dynamics()


class DynamicCityDeliveryEnv:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(self.graph.nodes)

        self.start_node_idx = None
        self.task_nodes_idx = None
        self.end_node_idx = None

        self.v_idx, self.d_idx = 0, 0
        self.cv_render = None

    @property
    def start_node(self):
        return self.nodes[self.start_node_idx]

    @property
    def end_node(self):
        return self.nodes[self.end_node_idx]

    def get_all_nodes(self):
        return [self.start_node_idx, ] + self.task_nodes_idx + [self.end_node_idx, ]

    def add_dynamics(self, dynamic):
        for u, v, data in self.graph.edges(data=True):
            key = '{}-{}'.format(u, v)
            times = dynamic[key]
            length = data.get('length', 1)
            traffic_time = length / (base_speed * times)
            data['times'] = times
            data['dynamic_weight'] = traffic_time

    def reset(self, reuse=False, render=False):
        if not reuse or self.task_nodes_idx is None:
            scenario = voo_scenarios[self.v_idx]
            dynamic = dynamics[self.d_idx]

            self.start_node_idx = scenario['start']
            self.end_node_idx = scenario['end']
            self.task_nodes_idx = scenario['tasks']
            self.add_dynamics(dynamic)

            self.v_idx = (self.v_idx + 1) % len(voo_scenarios)
            self.d_idx = (self.d_idx + 1) % len(dynamics)

        print('- Scenario Info: ')
        print('\t    Scenario ID: ', '{}_{}'.format(self.v_idx, self.d_idx))
        print('\t Start Node Idx: ', self.start_node_idx)
        print('\t   End Node Idx: ', self.end_node_idx)
        print('\t Task Nodes Idx: ', self.task_nodes_idx)

        if render and self.cv_render is None:
            self.cv_render = StaticRender(self.graph)

    def __build_task_graph(self, solver):
        all_nodes_idx = [self.start_node_idx, ] + self.task_nodes_idx + [self.end_node_idx, ]
        nodes_task_graph = [self.nodes[node] for node in all_nodes_idx]
        print('\t       Task Nodes: ', len(nodes_task_graph), nodes_task_graph)

        path_dict = {}
        dist_matrix = np.zeros((len(nodes_task_graph), len(nodes_task_graph)))
        for i, u in enumerate(nodes_task_graph):
            for j, v in enumerate(nodes_task_graph):
                if i == j: continue

                path, cost, done = solver.run(u, v)
                if not done: return None, None, None

                dist_matrix[i][j] = cost
                key = '{}-{}'.format(u, v)
                path_dict[key] = path

        return dist_matrix, nodes_task_graph, path_dict

    def run(self, voo_solver, opp_solver):
        start_time = time.time()
        matrix, nodes_task_graph, path_dict = self.__build_task_graph(opp_solver)

        voo_solver.reset(start=self.start_node_idx,
                         end=self.end_node_idx,
                         tasks=self.task_nodes_idx)
        vo, cost = voo_solver.run(matrix)
        print('\t   Visiting Order: ', len(nodes_task_graph), vo)
        assert vo[0] == self.start_node and vo[-1] == self.end_node
        delta_time = time.time() - start_time

        path = []
        for i, node1 in enumerate(vo[:-1]):
            key = '{}-{}'.format(node1, vo[i + 1])
            path_segment = path_dict[key]
            path += path_segment if i == 0 else path_segment[1:]
        print('\t             Cost: ', round(cost, 2))
        print('\t Time Consumption: ', round(delta_time, 3))
        print('\t     Dynamic Path: ', len(path), path)
        return cost, delta_time, vo, path

    def render(self, *args, **kwargs):
        if self.cv_render is None: return
        self.cv_render.draw(*args, **kwargs)


def main():
    np.random.seed(1111)

    place = "南京航空航天大学(将军路校区)"
    num_episodes = len(voo_scenarios) * len(dynamics)
    graph, p, num_action = load_graph(place_name=place)
    env = DynamicCityDeliveryEnv(graph)

    dp_voo_solver = DPTSPSolver(graph)
    dij_opp_solver = DijkstraPathPlanner(graph)

    # ga_opp_solver = DijkstraPathPlanner(graph)
    # rrt_opp_solver = DijkstraPathPlanner(graph)

    # rls_voo_solver = RLStaticTSPSolver(graph)
    # rls_opp_solver = RLStaticPathPlanner(graph, p, num_action)

    # rld_voo_solver = RLDynamicTSPSolver(graph)
    # rld_opp_solver = RLDynamicPathPlanner(graph, p, num_action)

    for episode in range(1, num_episodes + 1):
        print('----------------------------------')
        print('Episode: {}/{}: '.format(episode, num_episodes), end=' ')
        print('[Environment] Initialize environment')

        # DP + Dijkstra
        env.reset(reuse=False, render=True)
        print('- DP + Dijkstra: ')
        cost_1, time_1, vo_1, path_1 = env.run(voo_solver=dp_voo_solver, opp_solver=dij_opp_solver)
        env.render(path_1, vo_1)

        # GA + RRT
        # env.reset(reuse=True)
        # env.run(voo_solver=ga_opp_solver, opp_solver=rrt_opp_solver)

        # RL_static + RL_static
        # env.reset(reuse=True)
        # print('- RL (Static) + Dijkstra (Static): ')
        # cost_3, time_3, *_ = env.run(voo_solver=rls_voo_solver,
        #                              opp_solver=rls_opp_solver)

        # RL_dynamic + RL_dynamic
        # env.reset(reuse=True)
        # env.run(voo_solver=rl_voo_solver, opp_solver=rl_opp_solver)

        if episode >= 1: break


if __name__ == '__main__':
    main()
