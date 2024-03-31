import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def resolve_tsp(distance_matrix: pd.DataFrame, time_limit: int):
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

    search_parameters.time_limit.seconds = time_limit

    solution = routing.SolveWithParameters(search_parameters)

    result = get_solution(manager, routing, solution)
    mapping = dict(zip(range(distance_matrix.shape[0]), distance_matrix.index))
    path = [mapping.get(i) for i in result]
    return path
