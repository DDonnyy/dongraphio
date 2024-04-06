import networkit as nk
import networkx as nx
import pandas as pd


def get_subgraph(G_nx: nx.MultiDiGraph, attr, value):
    return G_nx.edge_subgraph([(u, v, k) for u, v, k, d in G_nx.edges(data=True, keys=True) if d[attr] in value])


def get_nx2nk_idmap(G_nx: nx.Graph) -> dict[int, int]:
    idmap = dict((id, u) for (id, u) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
    return idmap


def get_dist_matrix(
    graph: nx.DiGraph | nx.Graph, nodes_from: [], nodes_to: [], path_matrix=False, weight: str = "length_meter"
) -> (pd.DataFrame, pd.DataFrame | None):
    def reverse_dict(dictionary):
        return {v: k for k, v in dictionary.items()}

    mapping = get_nx2nk_idmap(graph)

    nk_graph = nk.nxadapter.nx2nk(graph, weight)

    distance_matrix = pd.DataFrame(0, index=nodes_from, columns=nodes_to).astype(object)
    if path_matrix:
        route_matrix = pd.DataFrame(index=nodes_from, columns=nodes_to)
        re_mapping = reverse_dict(mapping)
        for source in nodes_from:
            for dest in nodes_to:
                biDij = nk.distance.BidirectionalDijkstra(nk_graph, mapping.get(source), mapping.get(dest)).run()
                total_distance = biDij.getDistance()
                distance_matrix.loc[source, dest] = total_distance
                path = biDij.getPath()
                path = [re_mapping.get(x) for x in path]
                path.insert(0, source)
                path.append(dest)
                route_matrix.loc[source, dest] = path
                del biDij
        return distance_matrix, route_matrix

    spsp = nk.distance.SPSP(nk_graph, sources=[mapping.get(x) for x in nodes_from])
    spsp.setTargets(targets=[mapping.get(x) for x in nodes_to])
    spsp.run()
    for source in nodes_from:
        for dest in nodes_to:
            total_distance = spsp.getDistance(mapping.get(source), mapping.get(dest))
            distance_matrix.loc[source, dest] = total_distance
    del spsp
    return distance_matrix


def get_dist_matrix_for_tsp(graph: nx.DiGraph, route_nodes: list[tuple]) -> (pd.DataFrame, pd.DataFrame):
    route_nodes_ind = [x[0] for x in route_nodes]
    distance_matrix, route_matrix = get_dist_matrix(graph, route_nodes_ind, route_nodes_ind, True)
    mean_value = distance_matrix.values.mean()
    for i in route_nodes:
        node_1, n1_1, n2_1 = i
        for j in route_nodes:
            node_2, n1_2, n2_2 = j
            if (n1_1, n2_1) == (n2_2, n1_2):
                distance_matrix.loc[node_1, node_2] = (mean_value+distance_matrix.loc[node_1, node_2])/3
    return distance_matrix, route_matrix
