def inp():
    first_line = input().split(" ")
    num_points, num_trails = int(first_line[0]), int(first_line[1])
    adj_lst = {i: set() for i in range(num_points)}
    trail_len = {}
    trail_len_duplicate_count = {}
    for i in range(num_trails):
        trail = input().split(" ")
        node1, node2, length = int(trail[0]), int(trail[1]), int(trail[2])
        if node1 != node2:
            adj_lst[node1].add(node2)
            adj_lst[node2].add(node1)
            key = frozenset((node1, node2))
            if key in trail_len and length >= trail_len[key]:
                trail_len_duplicate_count[key] += 1 if length == trail_len[key] else 0
            else:
                trail_len[key] = length
                trail_len_duplicate_count[key] = 1
    return num_points, adj_lst, trail_len, trail_len_duplicate_count


def main():
    num_points, adj_lst, trail_len, trail_len_duplicate_count = inp()
    # print(adj_lst)
    # print(trail_len)
    # print(trail_len_duplicate_count)
    shortest_path = sum(trail_len.values())
    flower_path = set(trail_len.keys())

    <vul/>def dfs_recur(current_node, path):</vul>
        # print(path)
        nonlocal shortest_path, flower_path
        if current_node == num_points - 1:
            edges = [frozenset((path[i], path[i+1])) for i in range(len(path) - 1)]
            length = sum(trail_len[edge] for edge in edges)
            if length < shortest_path:
                flower_path = set(edges)
                shortest_path = length
            elif length == shortest_path:
                flower_path = flower_path.union(edges)
        else:
            for node in adj_lst[current_node]:
                <vul/>if node not in path:</vul>
                    path.append(node)
                    <vul/>dfs_recur(node, path)</vul>
                    path.pop()

    <vul/>dfs_recur(0, [0])</vul>
    # print(flower_path)
    return sum(trail_len[path] * trail_len_duplicate_count[path] for path in flower_path) * 2

if __name__ == '__main__':
    print(main())
