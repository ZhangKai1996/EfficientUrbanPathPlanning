import heapq


def dijkstra_path(graph, start_node, end_node, weight_key='dynamic_weight'):
    # 最小堆，存储 (累计成本, 当前节点, 路径)
    heap = [(0, start_node, [start_node])]
    visited = set()

    while heap:
        cost, current, path = heapq.heappop(heap)
        if current == end_node: return cost, path
        if current in visited: continue
        visited.add(current)

        for neighbor in graph[current].keys():
            if neighbor in visited: continue
            weight = graph[current][neighbor][0][weight_key]
            heapq.heappush(heap, (cost + weight, neighbor, path + [neighbor]))

    return float('inf'), []  # 如果没有路径
