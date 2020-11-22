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
                    score = random.random()
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

def do_matching_double_matches(graph):
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


    model = Model("Allow double matches based")
    edge_vars = {e:model.add_var(var_type = BINARY) for e in E}
    undirected = dict()
    for e in E:
        undirected[e] = edge_vars[e]
        undirected[e[1], e[0]] = edge_vars[e]

    is_best = {e: model.add_var(var_type=BINARY) for e in undirected.keys()}
    penalty = {v :model.add_var(var_type = BINARY) for v in V}
    happiness = {v:model.add_var() for v in V}


    epsilon = .01
    C = 1e3

    for v in V:
        model += xsum(edge_vars[s, t] for s, t in E if v in [s, t]) <= 1 + penalty[v]
        model += happiness[v] <= xsum(edge_vars[s, t] for s, t in E if v in [s, t]) * C
        if outputs[v]:
            model += xsum(is_best[(v, m)] for m in outputs[v]) == 1
            for m in outputs[v]:
                model += happiness[v] <= undirected[(v, m)] * weights[(v, m)] + (1 - is_best[(v, m)]) * C
        else:
            happiness[v] <= 0
    model.objective = maximize(xsum(happiness[v] for v in V) - epsilon * xsum(penalty[v] for v in V))
    model.optimize(max_seconds = 300)

    model.write("out_lp.lp")

    return sorted([e for e in E if edge_vars[e].x > .01])

def analyze(result, name = "matching", detailed = False):
    input(">>> See Results for " + name)
    scores = dict()
    for u in graph:
        for v, weight in graph[u]:
            scores[(int(u), int(v))] = weight


    print("Raw couples:", result)
    double_matches = 0
    mapping_dict = dict()
    for u, v in result:
        new_score = scores[(u, v)]
        if u in mapping_dict:
            double_matches += 1
            old = mapping_dict[u]
            old_score = scores[(u, old)]
            if old_score < new_score:
                mapping_dict[u] = v
        else:
            mapping_dict[u] = v

        if v in mapping_dict:
            double_matches += 1
            old = mapping_dict[v]
            old_score = scores[(v, old)]
            if old_score < new_score:
                mapping_dict[v] = u
        else:
            mapping_dict[v] = u

        if u == v:
            print("Somebody is partnered with themselves")
            exit(1)

        if str(u) not in graph or str(v) not in graph:
            print((u, v), "Nonexistent couple")
            exit(1)

    score_dict = scores
    scores = []

    print("Number of matched people", len(mapping_dict))
    print("Number of double matches", double_matches)

    for v in graph:
        person = mapping_dict.get(int(v), "")
        if detailed:
            print(v, "is matched with", str(person) + ("Nobody") * (1 - int(person is not "")))
        if person is not "":
            score = score_dict[(int(v), int(person))]
            if detailed:
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


graph = get_graph(random=True, N = 3000, AVG_DEGREE= 4)
# graph = get_stable_roommates_instance() # solution {1, 6}, {2,4}, {3, 5}}
result_max = do_matching(graph, visualize = False)
result_stable = do_matching_stable(graph, visualize = False, individual = 1, communal = 1000)
result_double = do_matching_double_matches(graph)
analyze(result_max, name = "Maximum Matching")
analyze(result_stable, name = "Stable pairings")
analyze(result_double, name = "Allow double matches")






