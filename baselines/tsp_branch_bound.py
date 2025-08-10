import math
import heapq


# 计算两城市间的距离
def calculate_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# 计算一个路径的总长度
def calculate_total_distance(path, distance_matrix):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i]][path[i + 1]]
    return total_distance


# 计算当前状态的界限（Bound）
def calculate_bound(dist_matrix, path, visited):
    n = len(dist_matrix)
    bound = 0
    # 计算行的最小值
    for i in range(n):
        if visited[i]:
            continue
        row_min = float('inf')
        for j in range(n):
            if not visited[j]:
                row_min = min(row_min, dist_matrix[i][j])
        if row_min != float('inf'):
            bound += row_min

    # 计算列的最小值
    for j in range(n):
        if visited[j]:
            continue
        col_min = float('inf')
        for i in range(n):
            if not visited[i]:
                col_min = min(col_min, dist_matrix[i][j])
        if col_min != float('inf'):
            bound += col_min

    return bound


# 分支定界法的核心函数
def branch_and_bound(dist_matrix, start=0, end=-1):
    n = len(dist_matrix)
    if end == -1: end = n-1

    # 最优解初始化为无穷大
    best_cost = float('inf')
    best_path = []

    # 使用优先队列管理待扩展的节点
    # 每个节点包含当前路径、当前费用、已访问城市的集合和界限
    pq = []

    # 初始化起点
    visited = [False] * n
    visited[start] = True  # 从指定的起点出发
    heapq.heappush(pq, (0, [start], visited, 0))  # (当前成本, 当前路径, 已访问状态, 当前界限)

    # 分支定界法
    while pq:
        # 从队列中弹出当前节点
        current_cost, current_path, current_visited, current_bound = heapq.heappop(pq)

        # 如果当前路径已经超过最优解，则剪枝
        if current_cost + current_bound >= best_cost:
            continue

        # 判断当前路径是否已经访问了所有城市
        if len(current_path) == n:
            # 如果路径已经包含所有城市，且以终点结束
            if current_path[-1] == end:
                total_cost = current_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = current_path
            continue

        # 进行分支扩展，考虑每个城市作为下一站
        for i in range(n):
            if not current_visited[i]:
                new_cost = current_cost + dist_matrix[current_path[-1]][i]
                new_path = current_path + [i]
                new_visited = current_visited[:]
                new_visited[i] = True

                # 计算当前状态的界限
                new_bound = calculate_bound(dist_matrix, new_path, new_visited)

                # 如果新路径的界限小于最优解，则加入队列
                if new_cost + new_bound < best_cost:
                    heapq.heappush(pq, (new_cost, new_path, new_visited, new_bound))

    return best_cost, best_path