import networkx as nx

from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, maximize, BINARY
import json
import numpy as np

import matplotlib.pyplot as plt

def get_graph(random=False, N = 100, AVG_DEGREE = 9):
    if not random:
        with open("possibleMatches.json", "r") as fh:
            return json.load(fh)
    else:
        import random
        out = {str(i):[] for i in range(N)}
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < AVG_DEGREE / N:
                    score = random.choice([4,4,4,4,4,4,4,4,4,4,6,9])
                    out[str(i)].append((j, score))
                    out[str(j)].append((i, score))

        return out
def do_matching(graph, visualize = True):
    print("Starting model")
    weights = dict()
    graph = {int(key): graph[key] for key in graph}
    E = set()
    V = graph.keys()

    for v in V:
        original = v
        for u, weight in graph[original]:
            s, t = (u, v) if u < v else (v, u)
            edge = (s,t)
            E.add(edge)
            weights[edge] = weight

    if visualize:
        graph = nx.Graph()
        graph.add_nodes_from(V)
        graph.add_edges_from(E)
        nx.draw_kamada_kawai(graph)
        plt.show()

    model = Model("Maximum matching")
    edge_vars = {e:model.add_var(var_type = BINARY) for e in E}
    for v in V:
        model += xsum(edge_vars[s, t] for s, t in E if v in [s, t]) <= 1
    model.objective = maximize(xsum(weights[edge] * edge_vars[edge] for edge in E))
    model.optimize(max_seconds = 300)
    return sorted([e for e in E if edge_vars[e].x > .01])

graph = get_graph(random=True, N = 5000, AVG_DEGREE= 25)
result = do_matching(graph, visualize = False)
print("Number of couples:", len(result))
print("Raw couples:", result)
mapping_dict = dict()
for u, v in result:
    if u in mapping_dict or v in mapping_dict:
        print("Invalid matching", u, v, mapping_dict.get(u, None), mapping_dict.get(v, None))
        exit(1)
    if u == v:
        print("Somebody is partnered with themselves")
        exit(1)
    if str(u) not in graph or str(v) not in graph:
        print((u, v), "Nonexistent couple")
        exit(1)
    mapping_dict[u] = v
    mapping_dict[v] = u

scores = []

for v in graph:
    person =  mapping_dict.get(int(v), "")
    print(v, "is matched with", str(person) + ("Nobody") * (1 - int(person is not "")))
    if person is not "":
        score = max(graph[str(v)], key = lambda lst: lst[0] == person)[1]
        print("This matching scores:", score)
        scores.append(score)

print("The number of people with a match:", len(scores))
print("Average score: ", np.mean(scores) )
print("Standard deviation: ", np.std(scores) )




