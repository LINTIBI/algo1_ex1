#Koral Sereb 207972282#
#Lin Tibi 318232139#

from collections import defaultdict
import sys
class Heap:

    def __init__(self):
        self.array = []
        self.size = 0
        self.index = []

    def swap_item(self, a, b):
        temp = self.array[a]
        b = int(b)
        a = int(a)
        self.array[a] = self.array[b]
        self.array[b] = temp

    def decrease_key(self, v, current):
        i = self.index[v]
        self.array[i][1] = current

        while i > 0 and self.array[int(i)][1] < self.array[int((i - 1) / 2)][1]:
            i = int(i)
            self.index[self.array[i][0]] = int((i - 1) / 2)
            self.index[self.array[int((i - 1) / 2)][0]] = i
            self.swap_item(i, (i - 1) / 2)
            i = (i - 1) / 2

    def is_into_min_heap(self, v):
        return self.index[v] < self.size

    def order_min_heapify(self, current):
        smallest = current
        leftChild = 2 * current + 1
        rightChild = 2 * current + 2

        if leftChild < self.size and self.array[leftChild][1] < self.array[smallest][1]:
            smallest = leftChild

        if rightChild < self.size and self.array[rightChild][1] < self.array[smallest][1]:
            smallest = rightChild

        if smallest != current:
            self.index[self.array[smallest][0]] = current
            self.index[self.array[current][0]] = smallest
            self.swap_item(smallest, current)

            self.order_min_heapify(smallest)

    def extract_min(self):

        if self.is_empty():
            return

        self.size -= 1
        root = self.array[0]

        last_item = self.array[self.size]
        self.array[0] = last_item

        self.index[last_item[0]] = 0
        self.index[root[0]] = self.size

        self.order_min_heapify(0)

        return root

    def is_empty(self):
        return self.size == 0

    @staticmethod
    def min_heap_item(v, weight):
        return [v, weight]


class Edge:
    def __init__(self, source, destination, weight):
        self.source = source
        self.destination = destination
        self.weight = weight

    def __str__(self):
        return "between " + str(self.source) + " to " + str(self.destination) + " ,weight: " + str(self.weight)


class Graph:
    def __init__(self, V):
        self.V = V
        self.graph = defaultdict(list)
        self.edges = []

    def add_edge(self, src, destination, weight):
        new_item = [destination, weight]
        self.graph[src].insert(0, new_item)

        new_item = [src, weight]
        self.graph[destination].insert(0, new_item)
        self.edges.append(Edge(src, destination, weight))

    def prim_mst(self):
        V = self.V
        key = []
        parent = []
        min_heap = Heap()
        min_heap.size = V

        for v in range(V):
            parent.append(-1)
            key.append(sys.maxsize)
            min_heap.array.append(min_heap.min_heap_item(v, key[v]))
            min_heap.index.append(v)

        min_heap.index[0] = 0
        key[0] = 0
        min_heap.decrease_key(0, key[0])

        while not min_heap.is_empty():

            new_heap_node = min_heap.extract_min()
            u = new_heap_node[0]

            for p in self.graph[u]:
                v = p[0]

                if min_heap.is_into_min_heap(v) and p[1] < key[v]:
                    key[v] = p[1]
                    parent[v] = u
                    min_heap.decrease_key(v, key[v])

        edges = []
        for i in range(1, len(parent)):
            edges.append([i, parent[i]])

        graph = defaultdict(list)

        for edge in edges:
            a, b = edge[0], edge[1]

            graph[a].append(b)
            if b != -1:
                graph[b].append(a)
        return graph

    def __str__(self):
        graph_string = ""
        for edge in self.edges:
            graph_string += edge.__str__() + "\n"
        return graph_string


def bfs(graph, source, destination):
    explored = []
    queue = [[source]]

    if source == destination:
        return

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node not in explored:
            neighbours = graph[node]

            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

                if neighbour == destination:
                    explored.append(node)
                    return explored
            explored.append(node)
    return explored


def exe1(graph, edge_list):
    for edge in edge_list:
        graph.add_edge(edge.source, edge.destination, edge.weight)
    print(graph)

    tree_graph = graph.prim_mst()

    print("results exe 1: ")
    for keys, values in tree_graph.items():
        print("from to")
        print(keys, values)

    return graph, tree_graph


def exe2(tree_graph, new_edge, graph):
    visited_arr = bfs(tree_graph, new_edge.source, new_edge.destination)
    max_edge = Edge(0, 0, 0)
    for edge in graph.edges:
        if edge.source in visited_arr and edge.destination in visited_arr:
            if edge.weight > max_edge.weight:
                max_edge = edge

    if max_edge.weight > new_edge.weight:
        tree_graph[max_edge.source].remove(max_edge.destination)
        tree_graph[max_edge.destination].remove(max_edge.source)
        tree_graph[new_edge.destination].append(new_edge.source)
        tree_graph[new_edge.source].append(new_edge.destination)

    print("results exe 2: ")
    for keys, values in tree_graph.items():
        print("from to")
        print(keys, values)


if __name__ == "__main__":
    graph = Graph(20)
    edge_list = []
    for x in range(3):
        for y in range(x + 1, 20):
            edge_list.append(Edge(x, y, x + y))

    print("exe 1")
    graph, tree_graph = exe1(graph, edge_list)

    print("exe 2 - will not change")
    new_edge = Edge(1, 2, 1000)
    print(new_edge)
    exe2(tree_graph, new_edge, graph)

    print("exe 2 - will change")
    new_edge = Edge(2, 3, 1)
    print(new_edge)
    exe2(tree_graph, new_edge, graph)