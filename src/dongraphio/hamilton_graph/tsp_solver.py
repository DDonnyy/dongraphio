import networkx as nx
import pandas as pd
import geopandas as gpd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from shapely import LineString
from shapely.ops import transform


def resolve_tsp(distance_matrix: pd.DataFrame):
    def get_solution(manager, routing, solution) -> list[int]:
        result = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            result.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return result

    dm = distance_matrix.values.tolist()
    dm = [[round(num) for num in sublist] for sublist in dm]
    manager = pywrapcp.RoutingIndexManager(len(dm), 1, 0)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dm[from_node][to_node]

    routing = pywrapcp.RoutingModel(manager)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)

    result = get_solution(manager, routing, solution)
    mapping = dict(zip(range(distance_matrix.shape[0]), distance_matrix.index))
    path = [mapping.get(i) for i in result]
    return path


def subgraph_by_path(path: list, path_matrix: pd.DataFrame, graph: nx.Graph) -> nx.Graph:
    graph_by_path = nx.DiGraph()
    def round_coordinates(
        x,
        y,
    ):
        return round(x, 2), round(y, 2)

    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    path_edges.append((path[-1], path[0]))
    route_number = str(path_edges[0][0]).split("_")[1].split(".")[0]
    for u, v in path_edges:
        route_ = path_matrix.loc[u, v]
        new_route_geometry = None
        for i in range(len(route_) - 1):
            node1 = str(route_[i])
            node2 = str(route_[i + 1])
            cur_geom = transform(round_coordinates, LineString(graph[node1][node2]["geometry"]))
            if new_route_geometry is None:
                new_route_geometry = cur_geom
            else:
                new_route_geometry = new_route_geometry.union(cur_geom)
        route_len = new_route_geometry.length
        graph_by_path.add_node(u, x=graph.nodes[u]["x"], y=graph.nodes[u]["y"], route=route_number)
        graph_by_path.add_node(v, x=graph.nodes[v]["x"], y=graph.nodes[v]["y"], route=route_number)
        graph_by_path.add_edge(u, v, geometry=new_route_geometry, route=route_number, length_meter=route_len)
    return graph_by_path