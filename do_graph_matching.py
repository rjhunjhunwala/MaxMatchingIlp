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
                    score = int(random.random() * 100) + 1
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
            weights[original, u] = weight


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
    model.objective = maximize(xsum( xsum(((weights[edge] + weights[edge[1], edge[0]]) / 2) * edge_vars[edge] for edge in E) for edge in E))
    model.optimize(max_seconds = 300)
    return sorted([e for e in E if edge_vars[e].x > .01])

def do_matching_stable(graph, visualize = True, individual = 1, communal = 10000000):
    print("Starting model")
    weights = dict()
    graph = {int(key): graph[key] for key in graph}
    E = set()
    V = graph.keys()
    inputs = {v:[] for v in V}
    outputs = {v:[] for v in V}
    for v in V:
        original = v
        for u, weight in graph[original]:
            s, t = (u, v) if u < v else (v, u)
            edge = (s,t)
            E.add(edge)
            weights[(original, u)] = weight
            outputs[original].append(u)
            inputs[u].append(original)

    if visualize:
        graph = nx.Graph()
        graph.add_nodes_from(V)
        graph.add_edges_from(E)
        nx.draw_kamada_kawai(graph)
        plt.show()

    model = Model("Rogue Couples based")
    edge_vars = {e:model.add_var(var_type = BINARY) for e in E}
    undirected = dict()
    for e in E:
        undirected[e] = edge_vars[e]
        undirected[e[1], e[0]] = edge_vars[e]
    rogue_vars = {e: model.add_var(var_type=BINARY) for e in E}
    partners = dict()
    for v in V:
        partners[v] = model.add_var()
        partners[v] = xsum(edge_vars[s, t] for s, t in E if v in [s, t])
        model += partners[v] <= 1
    for (u, v), rogue_var in rogue_vars.items():
        v_primes = [vp for vp in outputs[u] if weights[(u , vp)] < weights[(u, v)]]
        u_primes = [up for up in outputs[v] if weights[(v, up)] < weights[(v, u)]]

        model += 1 - partners[v] - partners[u] + xsum(undirected[u, vp] for vp in v_primes) + xsum(undirected[up, v] for up in u_primes) <= rogue_var



    model.objective = maximize(individual * xsum(((weights[edge] + weights[edge[1], edge[0]]) / 2) * edge_vars[edge] for edge in E) - communal * xsum(rogue_vars[edge] for edge in E))
    model.optimize(max_seconds = 300)
    return sorted([e for e in E if edge_vars[e].x > .01])

def analyze(result, name = "matching"):
    input(">>> See Results for " + name)
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
        person = mapping_dict.get(int(v), "")
        print(v, "is matched with", str(person) + ("Nobody") * (1 - int(person is not "")))
        if person is not "":
            score = (max(graph[str(v)], key=lambda lst: lst[0] == person)[1] + max(graph[str(person)], key=lambda lst: lst[0] == v)[1])/2
            print("This matching scores:", score)
            scores.append(score)

    print("The number of people with a match:", len(scores))
    print("Average score: ", np.mean(scores))
    print("Standard deviation: ", np.std(scores))

def get_stable_roommates_instance():
    """
    The following are the preference lists for a Stable Roommates instance involving 6 participants: 1, 2, 3, 4, 5, 6.

1 :   3   4   2   6   5
2 :   6   5   4   1   3
3 :   2   4   5   1   6
4 :   5   2   3   6   1
5 :   3   1   2   4   6
6 :   5   1   3   4   2
    :return: 
    """

    return {"1" :  [(3, 6),   (4, 5),   (2, 4),   (6, 3),  (5, 2)],
"2" :   [(6,6),  (5,5),  (4,4),  (1,3) , (3, 2)],
"3" :   [(2, 6), (4, 5), (5, 4),  (1,3), (6, 2)],
"4" :   [(5, 6), (2, 5), (3, 4), (6, 3), (1, 2)],
"5" :   [(3, 6), (1, 5), (2, 4), (4 ,3), (6, 2)],
"6" :   [(5, 6),  (1, 5), (3, 4), (4, 3), (2, 2)]}


graph = get_graph(random=True, N = 300, AVG_DEGREE= 15)
# graph = get_stable_roommates_instance() # solution {1, 6}, {2,4}, {3, 5}}
result_max = do_matching(graph, visualize = False)
result_stable = do_matching_stable(graph, visualize = False, individual = 1, communal = 1000)
analyze(result_max, name = "Maximum Matching")
analyze(result_stable, name = "Stable pairings")





