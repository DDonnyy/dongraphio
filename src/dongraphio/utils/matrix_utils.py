import networkit as nk
import networkx as nx
import pandas as pd
import shapely.geometry
import shapely.wkt
from tqdm.auto import tqdm


def get_subgraph(G_nx: nx.Graph, attr, value):
    return G_nx.edge_subgraph([(u, v, k) for u, v, k, d in G_nx.edges(data=True, keys=True) if d[attr] in value])


def convert_nx2nk(G_nx: nx.Graph, idmap: dict[int, int] | None = None, weight: str | None = None):
    if not idmap:
        idmap = get_nx2nk_idmap(G_nx)
    n = max(idmap.values()) + 1
    edges = list(G_nx.edges())

    if weight:
        G_nk: nk.Graph = nk.Graph(n, directed=G_nx.is_directed(), weighted=True)
        for u_, v_ in edges:
            u, v = idmap[u_], idmap[v_]
            d = dict(G_nx[u_][v_])
            u_ = int(u_)
            v_ = int(v_)
            if len(d) > 1:
                for d_ in d.values():
                    v__ = G_nk.addNodes(2)
                    u__ = v__ - 1
                    w = round(d_[weight], 1) if weight in d_ else 1
                    G_nk.addEdge(u, v, w)
                    G_nk.addEdge(u_, u__, 0, addMissing=True)
                    G_nk.addEdge(v_, v__, 0, addMissing=True)
            else:
                d_ = list(d.values())[0]
                w = round(d_[weight], 1) if weight in d_ else 1
                G_nk.addEdge(u, v, w)
    else:
        G_nk = nk.Graph(n, directed=G_nx.is_directed())
        for u_, v_ in edges:
            u, v = idmap[u_], idmap[v_]
            G_nk.addEdge(u, v)

    return G_nk


def load_graph_geometry(G_nx: nx.Graph, node: bool = True, edge: bool = False) -> nx.Graph:
    if edge:
        for _u, _v, data in G_nx.edges(data=True):
            data["geometry"] = shapely.wkt.loads(data["geometry"])
    if node:
        for _u, data in G_nx.nodes(data=True):
            data["geometry"] = shapely.geometry.Point([data["x"], data["y"]])
    return G_nx


def get_nx2nk_idmap(G_nx: nx.Graph) -> dict[int, int]:
    idmap = dict((id, u) for (id, u) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
    return idmap


def get_nk_distances(nk_dists, loc):  # ?
    target_nodes = loc.index
    source_node = loc.name
    distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]
    return pd.Series(data=distances, index=target_nodes)


def get_dist_matrix(
    graph: nx.Graph, nodes_from: [], nodes_to: [], path_matrix=False, weight: str = "length_meter"
) -> (pd.DataFrame, pd.DataFrame | None):
    graph = nx.DiGraph(graph)
    mapping = get_nx2nk_idmap(graph)
    nk_graph = nk.nxadapter.nx2nk(graph, weight)

    distance_matrix = pd.DataFrame(0, index=nodes_from, columns=nodes_to).astype(object)
    if path_matrix:
        route_matrix = pd.DataFrame(index=nodes_from, columns=nodes_to)
    for source in tqdm(nodes_from, total=len(nodes_from), desc="Calculating distance matrix"):
        for dest in nodes_to:
            biDij = nk.distance.BidirectionalDijkstra(nk_graph, mapping.get(source), mapping.get(dest), True).run()
            path = biDij.getPath()
            path.insert(0,source)
            path.append(dest)
            total_distance = biDij.getDistance()
            if path_matrix:
                route_matrix.loc[source, dest] = path
            distance_matrix.loc[source, dest] = total_distance
    if path_matrix:
        return distance_matrix, route_matrix
    return distance_matrix


def get_dist_matrix_for_tsp(graph: nx.Graph, route_nodes: list[tuple]) -> (pd.DataFrame, pd.DataFrame):
    route_nodes_ind = [x[0] for x in route_nodes]
    distance_matrix, route_matrix = get_dist_matrix(graph, route_nodes_ind, route_nodes_ind, True)
    max_value = distance_matrix.values.max()
    for i in route_nodes:
        node_1, n1_1, n2_1 = i
        for j in route_nodes:
            node_2, n1_2, n2_2 = j
            if (n1_1, n2_1) == (n2_2, n1_2):
                distance_matrix.loc[node_1, node_2] = max_value
    return distance_matrix, route_matrix
