

def held_karp(dist_matrix, start=0, end=None):
    n = len(dist_matrix)
    if end is None:
        end = n - 1

    C = {}  # 状态集合：key=(mask, last_node), value=(cost, path)
    for k in range(n):
        if k == start or k == end:
            continue
        mask = (1 << start) | (1 << k)
        C[(mask, k)] = (dist_matrix[start][k], [start, k])

    # 枚举子集大小
    for subset_size in range(3, n + 1):
        for mask in range(1 << n):
            if (mask & (1 << start)) == 0 or (mask & (1 << end)) != 0:
                continue
            if bin(mask).count("1") != subset_size:
                continue

            for u in range(n):
                if u == start or u == end or not (mask & (1 << u)):
                    continue

                prev_mask = mask ^ (1 << u)
                min_cost = float('inf')
                min_path = []

                for v in range(n):
                    if v == start or v == end or v == u or not (prev_mask & (1 << v)):
                        continue
                    if (prev_mask, v) not in C:
                        continue
                    prev_cost, prev_path = C[(prev_mask, v)]
                    cost = prev_cost + dist_matrix[v][u]
                    if cost < min_cost:
                        min_cost = cost
                        min_path = prev_path + [u]

                if min_path:
                    C[(mask, u)] = (min_cost, min_path)

    # 枚举所有以某节点 u 结尾的路径，然后加上 u -> end 的边
    min_total_cost = float('inf')
    best_total_path = []
    final_mask = ((1 << n) - 1) ^ (1 << end)  # 所有点都访问，但 end 除外

    for u in range(n):
        if u == start or u == end:
            continue
        if (final_mask, u) not in C:
            continue
        prev_cost, prev_path = C[(final_mask, u)]
        total_cost = prev_cost + dist_matrix[u][end]
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_total_path = prev_path + [end]

    return min_total_cost, best_total_path
