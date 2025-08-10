import time
import random

import numpy as np
import networkx as nx


class GeneticAlgorithmSolver:
    def __init__(self, dist_matrix, population_size=50, generations=100, mutation_rate=0.1):
        self.dist_matrix = dist_matrix          # 距离矩阵
        self.population_size = population_size  # 种群大小
        self.generations = generations          # 代数
        self.mutation_rate = mutation_rate      # 变异率
        self.num_nodes = len(dist_matrix)       # 节点数量

    def run(self, visiting_nodes, task_dict):
        print('\t Run the Generic Algorithm ...')
        n = self.num_nodes
        if n <= 3:
            best_individual = list(range(n))
            best_cost = sum(self.dist_matrix[n1][best_individual[i + 1]]
                            for i, n1 in enumerate(best_individual[:-1]))
            return best_individual, best_cost

        population = self.__generate_population()  # 初始种群

        best_individual = None
        best_fitness = float('-inf')
        for generation in range(self.generations):
            new_population = []
            # 精英策略：保存最好的个体
            for individual in population:
                fitness_value = self.__fitness(individual, visiting_nodes, task_dict)
                if fitness_value > best_fitness:
                    best_fitness = fitness_value
                    best_individual = individual
            # 生成新种群
            while len(new_population) < self.population_size:
                parent1, parent2 = self.__selection(population, visiting_nodes, task_dict)  # 选择两个父代
                child = self.__crossover(parent1, parent2)  # 交叉生成子代
                child = self.__mutate(child)  # 变异子代
                new_population.append(child)

            population = new_population  # 更新种群
        return best_individual, 1 / best_fitness  # 返回最优路径及其距离

    def __generate_population(self):
        n = self.num_nodes

        population = []
        for _ in range(self.population_size):
            individual = list(range(1, n-1))  # 初始路径从 0 到 n-1
            random.shuffle(individual)  # 随机打乱路径
            individual = [0, ] + individual + [n-1, ]
            population.append(individual)
        return population

    def __fitness(self, individual, visiting_nodes, task_dict):
        n = self.num_nodes

        new_visiting_nodes = [visiting_nodes[idx] for idx in individual]
        new_visiting_nodes = new_visiting_nodes[1:-1]

        is_available_individual = True
        for idx, node in enumerate(new_visiting_nodes):
            if node not in task_dict.keys(): continue

            task = task_dict[node]
            if task.is_picked(): continue
            p_idx = new_visiting_nodes.index(task.pickup_point)
            if p_idx > idx:
                is_available_individual = False

        times = 0.01 if is_available_individual else 1.0
        total_distance = 0
        for i in range(n-1):
            total_distance += self.dist_matrix[individual[i]][individual[i + 1]]
        return 1 / (total_distance * times) * 1000 # 适应度 = 1 / 路径总距离

    def __selection(self, population, visiting_nodes, task_dict):
        """
        选择操作：轮盘赌选择方法，根据适应度选择父代
        """
        fitness_values = [self.__fitness(ind, visiting_nodes, task_dict) for ind in population]
        total_fitness = sum(fitness_values)
        probabilities = [fit / total_fitness for fit in fitness_values]
        selected = random.choices(population, probabilities, k=2)  # 选择两个父代
        return selected

    def __crossover(self, parent1, parent2):
        """
        交叉操作：部分匹配交叉（PMX）生成新个体
        """
        n = self.num_nodes

        point1, point2 = sorted(random.sample(range(1, n-1), 2))  # 随机选择交叉点
        # 创建子代
        child = [None] * len(parent1)
        # 复制父代的交叉部分
        child[point1:point2 + 1] = parent1[point1:point2 + 1]
        # 填充其余部分
        for i in range(len(parent2)):
            if parent2[i] not in child:
                for j in range(len(child)):
                    if child[j] is None:
                        child[j] = parent2[i]
                        break
        return child

    def __mutate(self, individual):
        """
        变异操作：随机交换路径中的两个节点
        """
        n = self.num_nodes

        if random.random() < self.mutation_rate:
            i, j = random.sample(range(1, n-1), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual


class DynamicProgrammingSolver:
    def __init__(self, dist_matrix):
        self.dist_matrix = dist_matrix  # 访问节点之间的距离矩阵

    def run(self, visiting_nodes, task_dict):
        print('\t Run the Dynamic Programming ...')
        n = len(self.dist_matrix)

        start_node = 0    # 起点
        end_node = n - 1  # 终点

        # dp[mask][i]表示到达mask子集中的所有节点后，最后停在节点i的最短路径
        dp = np.inf * np.ones((1 << n, n))   # dp表格初始化为无穷大
        dp[1 << start_node][start_node] = 0  # 从起点开始

        # 遍历所有子集
        for mask in range(1 << n):
            for u in range(n):
                if (mask & (1 << u)) == 0: continue  # 如果u不在子集mask中，跳过
                for v in range(n):
                    if mask & (1 << v): continue     # 如果v已经在子集mask中，跳过

                    if visiting_nodes[v] in task_dict.keys():
                        task = task_dict[visiting_nodes[v]]
                        if not task.is_picked():
                            w = visiting_nodes.index(task.pickup_point)
                            if (mask & (1 << w)) == 0: continue  # 如果送货点之前没有取货点被访问，则跳过
                    dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + self.dist_matrix[u][v])

        # 最优路径长度是从起点到终点的最短路径
        mask = (1 << n) - 1  # 所有节点都访问过
        min_cost = dp[mask][end_node]  # 到达终点的最短路径成本

        # 回溯最优路径
        path = [end_node]
        last_node = end_node
        while mask != (1 << start_node):  # 直到回到起点
            for v in range(n):
                if mask & (1 << v):
                    if dp[mask][last_node] == dp[mask ^ (1 << last_node)][v] + self.dist_matrix[v][last_node]:
                        path.append(v)
                        mask ^= (1 << last_node)  # 移除last_node节点
                        last_node = v
                        break

        path = path[::-1]  # 因为回溯是从终点到起点，最后需要将路径反转
        return path, min_cost


class PathPlanner:
    def __init__(self, algo='DP', **kwargs):
        self.algo = algo
        self.kwargs = kwargs
        self.solver = self.__get_solver()

    def __get_solver(self, algo=None):
        if algo is None:
            algo = self.algo

        if algo == 'DP': return DynamicProgrammingSolver
        if algo == 'GA': return GeneticAlgorithmSolver
        raise NotImplementedError

    def plan_path(self, city_map, tasks, start_point, end_point):
        print('[Solver] Planning the path ...')
        print('\t Now position: ', start_point)

        start_time = time.time()

        all_goal_nodes = []
        task_dict = {}
        for task in tasks:
            if task.pickup_time is None:
                all_goal_nodes.append(task.pickup_point)
            if task.delivery_time is None:
                all_goal_nodes.append(task.delivery_point)
                task_dict[task.delivery_point] = task

        # 当前访问顺序，包括起点、取货点、送货点和最终目标点
        visiting_nodes = [start_point, ] + all_goal_nodes + [end_point, ]
        n = len(visiting_nodes)
        print("\t Visiting nodes:", n, visiting_nodes)

        dist_matrix = np.zeros((n, n))
        path_dict = {node: {} for node in visiting_nodes}
        for i in range(n):
            goal_node1 = visiting_nodes[i]
            for j in range(i + 1, n):
                goal_node2 = visiting_nodes[j]

                path = nx.shortest_path(city_map, source=goal_node1, target=goal_node2, weight='dynamic_weight')
                dist_matrix[i][j] = sum(
                    city_map[node1][path[k + 1]][0]['dynamic_weight']
                    for k, node1 in enumerate(path[:-1])
                )
                path_dict[goal_node1][goal_node2] = path

                path = nx.shortest_path(city_map, source=goal_node2, target=goal_node1, weight='dynamic_weight')
                dist_matrix[j][i] = sum(
                    city_map[node1][path[k + 1]][0]['dynamic_weight']
                    for k, node1 in enumerate(path[:-1])
                )
                path_dict[goal_node2][goal_node1] = path

        # Step 2: 使用动态规划/遗传算法求解TSP最优路径顺序
        best_order, min_cost = self.solver(dist_matrix, **self.kwargs).run(visiting_nodes, task_dict)
        new_visiting_nodes = [visiting_nodes[idx] for idx in best_order]
        is_available_path = True
        for task in tasks:
            if task.pickup_time is not None: continue

            assert task.delivery_time is None
            idx_p = new_visiting_nodes.index(task.pickup_point)
            idx_d = new_visiting_nodes.index(task.delivery_point)
            if idx_p > idx_d:
                is_available_path = False
        print('\t Best visiting order: ', best_order)
        print('\t New visiting nodes: ', new_visiting_nodes)
        print('\t Minimum cost: ', min_cost)
        print('\t PD constraint: ', is_available_path)

        # Step 3: 根据最优顺序扩展成完整的路径
        planned_path = [start_point, ]
        for i, n1 in enumerate(new_visiting_nodes[:-1]):
            n2 = new_visiting_nodes[i+1]
            shortest_path = path_dict[n1][n2]
            print('\t\t Segment {}: '.format(i+1), n1, n2, shortest_path)
            planned_path += shortest_path[1:]  # 跳过重复的起点
        return planned_path, min_cost, time.time() - start_time