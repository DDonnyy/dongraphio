from itertools import chain

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import LineString, Point, from_wkt, wkt
from shapely.ops import nearest_points, split, substring, transform


def parse_overpass_route_response(loc: dict, city_crs: int, boundary: gpd.GeoDataFrame, use_boundary: bool):
    route = pd.DataFrame(loc["members"])
    ways = route[route["type"] == "way"]
    if len(ways) > 0:
        ways = ways["geometry"].reset_index(drop=True)
        ways = ways.apply(pd.DataFrame)
        ways = ways.apply(lambda x: LineString(list(zip(x["lon"], x["lat"]))))
        ways = gpd.GeoDataFrame(ways.rename("geometry")).set_crs(4326)
        if ways.within(boundary).all() or not use_boundary:
            # fix topological errors and then make LineString from MultiLineString
            ways = _get_linestring(ways.to_crs(city_crs))
        else:
            ways = None
    else:
        ways = None

    if "node" in route["type"].unique():
        platforms = route[route["type"] == "node"][["lat", "lon"]].reset_index(drop=True)
        platforms = platforms.apply(lambda x: Point(x["lon"], x["lat"]), axis=1)
    else:
        platforms = None

    return pd.Series({"way": ways, "platforms": platforms})


def project_platforms(loc, city_crs):

    project_threshold = 5
    edge_indent = 10

    platforms = loc["platforms"]
    line = loc["way"]
    line_length = line.length

    if platforms is not None:
        platforms = gpd.GeoSeries(platforms).set_crs(4326).to_crs(city_crs)
        stops = platforms.apply(lambda x: nearest_points(line, x)[0])
        stops = gpd.GeoDataFrame(stops).rename(columns={0: "geometry"}).set_geometry("geometry")
        stops = _recursion(stops, project_threshold)

        check_reverse = gpd.GeoSeries([Point(line.coords[0]), Point(line.coords[-1])]).distance(stops[0]).idxmin()
        if check_reverse == 1:
            line = list(line.coords)
            line.reverse()
            line = LineString(line)

        stops_distance = stops.apply(line.project).sort_values()
        stops = stops.loc[stops_distance.index]
        condition = (stops_distance > edge_indent) & (stops_distance < line_length - edge_indent)
        stops, distance = stops[condition].reset_index(drop=True), [0] + list(stops_distance[condition])
        distance.append(line_length)

        if len(stops) > 0:
            start_line = gpd.GeoSeries(Point(line.coords[0])).set_crs(city_crs)
            end_line = gpd.GeoSeries(Point(line.coords[-1])).set_crs(city_crs)
            stops = pd.concat([start_line, stops, end_line]).reset_index(drop=True)
        else:
            stops, distance = _get_line_from_start_to_end(line, line_length)
    else:
        stops, distance = _get_line_from_start_to_end(line, line_length)

    pathes = [
        [
            tuple(round(c, 2) for c in stops[i].coords[0]),
            tuple(round(c, 2) for c in stops[i + 1].coords[0]),
        ]
        for i in range(len(stops) - 1)
    ]

    return pd.Series({"pathes": pathes, "distance": distance})


def get_nearest_edge_geometry(points, G):

    G = G.edge_subgraph([(u, v, n) for u, v, n, e in G.edges(data=True, keys=True) if e["type"] == "walk"])
    G = _convert_geometry(G.copy())
    coords = list(points.geometry.apply(lambda x: list(x.coords)[0]))
    x = [c[0] for c in list(coords)]
    y = [c[1] for c in list(coords)]
    edges, distance = ox.distance.nearest_edges(G, x, y, return_dist=True)
    edges_geom = list(map(lambda x: (x, G[x[0]][x[1]][x[2]]["geometry"]), edges))
    edges_geom = pd.DataFrame(edges_geom, index=points.index, columns=["edge_id", "edge_geometry"])
    edges_geom["distance_to_edge"] = distance
    return pd.concat([points, edges_geom], axis=1)


def project_point_on_edge(points_edge_geom):

    points_edge_geom["nearest_point_geometry"] = points_edge_geom.apply(
        lambda x: nearest_points(x.edge_geometry, x.geometry)[0], axis=1
    )
    points_edge_geom["len"] = points_edge_geom.apply(lambda x: x.edge_geometry.length, axis=1)
    points_edge_geom["len_from_start"] = points_edge_geom.apply(lambda x: x.edge_geometry.project(x.geometry), axis=1)
    points_edge_geom["len_to_end"] = points_edge_geom.apply(lambda x: x.edge_geometry.length - x.len_from_start, axis=1)

    return points_edge_geom


def update_edges(points_info, G):

    G_with_drop_edges = _delete_edges(points_info, G)  # pylint: disable=invalid-name
    updated_G, split_points = _add_splitted_edges(G_with_drop_edges, points_info)  # pylint: disable=invalid-name
    updated_G, split_points = _add_connecting_edges(updated_G, split_points)  # pylint: disable=invalid-name
    return updated_G, split_points


def _get_linestring(route):
    """
    get_linestring(route)
    Method for converting MultiLineString to LineString in any route
    """

    def find_equals_line(loc: dict, series: gpd.GeoSeries) -> list | None:

        series = series.drop(loc.name)
        eq_lines = series.geometry.apply(lambda x: x.equals_exact(loc.geometry, 1e-6))
        eq_lines = series[eq_lines].index

        equal_lines = sorted(list(eq_lines) + [loc.name]) if len(eq_lines) > 0 else None

        return equal_lines

    def find_connection(loc, df):
        df = df.drop(loc.name)
        bool_ser = df.intersects(loc.geometry)
        connect_lines = df[bool_ser].index

        if len(connect_lines) > 0:
            return list(connect_lines)
        return None

    def get_sequences(connect_ser: gpd.GeoSeries, sequences=None):

        if sequences is None:
            sequences = []
        num_con = connect_ser.apply(len)
        finite_points = pd.DataFrame(connect_ser[num_con == 1].rename("value"))

        if len(finite_points) == 0:
            return None

        sequence = move_next(finite_points.index[0], connect_ser, [])
        sequences.append(sequence)

        route_finite = finite_points.index.isin(sequence)
        if route_finite.all():
            return sequences
        connect_ser = connect_ser.drop(finite_points.index[route_finite])
        sequences = get_sequences(connect_ser, sequences)
        return sequences

    def move_next(path: str, series: gpd.GeoSeries, sequence: list, branching=None):

        sequence.append(path)
        try:
            series = series.drop(path)
        except Exception:  # seems not OK
            pass
        bool_next_path = series.apply(lambda x: path in x)
        next_path = series[bool_next_path].index

        if len(next_path) == 0:
            return sequence

        if len(next_path) > 1:

            if branching is None:
                branches_start = path
                sequence_variance = []
                for n_path in next_path:
                    series_ = series.drop([path_ for path_ in next_path if path_ != n_path])
                    sequence_ = move_next(n_path, series_, [], branches_start)
                    sequence_variance.append(sequence_)

            else:
                return sequence

            len_sequence = [len(sec) for sec in sequence_variance]
            max_sequence = len_sequence.index(max(len_sequence))
            sequence_ = sequence_variance[max_sequence]
            series_ = series.drop(list(chain(*[sec[-2:] for sec in sequence_variance])))
            sequence = sequence + sequence_
            sequence_ = move_next(sequence_[-1], series_, sequence_, None)
            return sequence + sequence_

        sequence = move_next(next_path[0], series, sequence, branching)
        return sequence

    equal_lines = route.apply(lambda x: find_equals_line(x, route), axis=1).dropna()
    lines_todel = list(chain(*[line[1:] for line in list(equal_lines)]))
    route = route.drop(lines_todel).reset_index()

    path_buff = gpd.GeoSeries(route.geometry.buffer(0.01))
    connect_series = route.apply(lambda x: find_connection(x, path_buff), axis=1).dropna()
    sequences = get_sequences(connect_series, [])
    if sequences is None:
        return None

    len_sequence = [len(sec) for sec in sequences]
    max_sequence = len_sequence.index(max(len_sequence))
    sequence = sequences[max_sequence]
    comlete_line = [route.geometry[sequence[0]]]

    for i in range(len(sequence) - 1):
        line1 = comlete_line[i]
        line2 = route.geometry[sequence[i + 1]]
        _con_point1, con_point2 = nearest_points(line1, line2)

        line2 = list(line2.coords)
        check_reverse = gpd.GeoSeries([Point(line2[0]), Point(line2[-1])]).distance(con_point2).idxmin()
        if check_reverse == 1:
            line2.reverse()

        comlete_line.append(LineString(line2))

    comlete_line = list(chain(*[list(line.coords) for line in comlete_line]))
    comlete_line = list(pd.Series(comlete_line).drop_duplicates())

    return LineString(comlete_line)


def _recursion(stops: gpd.GeoSeries, threshold):

    stops["to_del"] = stops.apply(lambda x: _get_index_to_delete(stops, x, threshold), axis=1)

    if stops["to_del"].isna().all():
        return stops["geometry"]
    stops_near_pair = stops.dropna().apply(lambda x: sorted([x.name, x.to_del]), axis=1)
    stops_to_del = [pair[0] for pair in stops_near_pair]
    stops = stops.drop(stops_to_del)
    stops = _recursion(stops, threshold)

    return stops.reset_index(drop=True)


def _get_index_to_delete(other_stops: gpd.GeoDataFrame, loc_stop: gpd.GeoSeries, threshold: int | float) -> int | None:

    stops_to_del = other_stops.geometry.distance(loc_stop.geometry).sort_values().drop(loc_stop.name)
    stops_to_del = list(stops_to_del[stops_to_del < threshold].index)

    if len(stops_to_del) > 0:
        return stops_to_del[0]
    return None


def _get_line_from_start_to_end(line, line_length: int | float):

    start_line = gpd.GeoSeries(Point(line.coords[0]))
    end_line = gpd.GeoSeries(Point(line.coords[-1]))
    stops = pd.concat([start_line, end_line]).reset_index(drop=True)
    distance = [0, line_length]

    return stops, distance


def _convert_geometry(graph):
    for _u, _v, _n, data in graph.edges(data=True, keys=True):
        data["geometry"] = wkt.loads(data["geometry"])
    return graph


def _delete_edges(project_points, G):

    bunch_edges = []
    G_copy = _convert_geometry(G.copy())  # pylint: disable=invalid-name
    for e in list(project_points["edge_id"]):
        flag = _check_parallel_edge(G_copy, *e)
        if flag == 2:
            bunch_edges.extend([(e[0], e[1], e[2]), (e[1], e[0], e[2])])
        else:
            bunch_edges.append((e[0], e[1], e[2]))

    bunch_edges = list(set(bunch_edges))
    G.remove_edges_from(bunch_edges)

    return G


def _check_parallel_edge(G, u, v, n) -> int:
    if u == v:
        return 1
    if G.has_edge(u, v) and G.has_edge(v, u):
        if G[u][v][n]["geometry"].equals(G[v][u][n]["geometry"]):
            return 2
        return 1
    return 1


def _add_splitted_edges(G, split_nodes):

    start_node_idx = max((G.nodes)) + 1
    split_nodes["node_id"] = range(start_node_idx, start_node_idx + len(split_nodes))
    nodes_bunch = split_nodes.apply(_generate_nodes_bunch, axis=1)
    nodes_attr = (
        split_nodes.set_index("node_id")
        .nearest_point_geometry.apply(
            lambda x: {
                "x": round(list(x.coords)[0][0], 2),
                "y": round(list(x.coords)[0][1], 2),
            }
        )
        .to_dict()
    )
    G.add_edges_from(list(nodes_bunch.explode()))
    nx.set_node_attributes(G, nodes_attr)

    return G, split_nodes


def _generate_nodes_bunch(split_point):

    edge_pair = []
    edge_nodes = split_point.edge_id
    edge_geom_ = split_point.edge_geometry
    new_node_id = split_point.node_id
    len_from_start = split_point.len_from_start
    len_to_end = split_point.len_to_end
    len_edge = split_point.len

    fst_edge_attr = {
        "length_meter": len_from_start,
        "geometry": str(substring(edge_geom_, 0, len_from_start)),
        "type": "walk",
    }
    snd_edge_attr = {
        "length_meter": len_to_end,
        "geometry": str(substring(edge_geom_, len_from_start, len_edge)),
        "type": "walk",
    }
    edge_pair.extend(
        [
            (edge_nodes[0], new_node_id, fst_edge_attr),
            (new_node_id, edge_nodes[0], fst_edge_attr),
            (new_node_id, edge_nodes[1], snd_edge_attr),
            (edge_nodes[1], new_node_id, snd_edge_attr),
        ]
    )

    return edge_pair


def _add_connecting_edges(G: nx.Graph, split_nodes: gpd.GeoDataFrame) -> tuple[nx.Graph, gpd.GeoDataFrame]:

    start_node_idx = split_nodes["node_id"].max() + 1
    split_nodes["connecting_node_id"] = list(range(start_node_idx, start_node_idx + len(split_nodes)))
    nodes_attr = (
        split_nodes.set_index("connecting_node_id")
        .geometry.apply(lambda p: {"x": round(p.coords[0][0], 2), "y": round(p.coords[0][1], 2)})
        .to_dict()
    )
    conn_edges = split_nodes.apply(
        lambda x: (
            x.node_id,
            x.connecting_node_id,
            {
                "type": "walk",
                "length_meter": round(x.distance_to_edge, 3),
                "geometry": str(LineString([x.geometry, x.nearest_point_geometry])),
            },
        ),
        axis=1,
    )
    conn_edges_another_direct = conn_edges.apply(lambda x: (x[1], x[0], x[2]))
    G.add_edges_from(conn_edges.tolist() + conn_edges_another_direct.tolist())
    nx.set_node_attributes(G, nodes_attr)
    return G, split_nodes


def nx_to_gdf(
    graph: nx.MultiDiGraph, nodes: bool = False, edges: bool = False
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    for _, _, data in graph.edges(data=True):
        data["geometry"] = from_wkt(str(data["geometry"]))

    if edges and nodes:
        gdf_graph_nodes, gdf_graph_edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        gdf_graph_edges = gdf_graph_edges.reset_index()
        return gdf_graph_nodes, gdf_graph_edges
    elif edges:
        gdf_graph_edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        gdf_graph_edges = gdf_graph_edges.reset_index()
        return gdf_graph_edges
    elif nodes:
        gdf_graph_nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)
        return gdf_graph_nodes
    raise ValueError("You must specify either nodes or edges as True.")


def buffer_directed_roads(graph_gdf: gpd.GeoDataFrame, buffer: int) -> gpd.GeoDataFrame:

    gdf_graph_edges_buffer = graph_gdf.copy()
    gdf_graph_edges_buffer["geometry"] = gdf_graph_edges_buffer["geometry"].apply(
        lambda x: LineString(x).buffer(-buffer, single_sided=True)
    )

    return gdf_graph_edges_buffer


def project_points_on_graph(
    graph_gdf_edges: gpd.GeoDataFrame, graph_gdf_edges_buffer: gpd.GeoDataFrame, points_to_project: gpd.GeoDataFrame
) -> pd.DataFrame:
    join = gpd.sjoin_nearest(points_to_project, graph_gdf_edges_buffer, distance_col="dist", how="inner")

    projected_points = pd.DataFrame(data=None)

    for _, row in join.iterrows():
        point: Point = row["geometry"]
        edges_to_project, start, end, key = row[["index_right", "u", "v", "key"]]
        line: LineString = graph_gdf_edges.loc[edges_to_project, "geometry"]
        nearest_point = line.interpolate(line.project(point))
        graph_data = (str(edges_to_project) + "_" + row["label"], start, end, key)
        projected_points = pd.concat(
            [projected_points, pd.DataFrame({"edge": [graph_data], "geometry": [nearest_point]})]
        )

    projected_points = projected_points.groupby("edge")["geometry"].apply(list).reset_index()
    return projected_points


def _remove_multiedges_except_key(multidigraph: nx.MultiDiGraph, node1, node2, key_to_save):
    edge_keys = list(multidigraph[node1][node2].keys())
    for key in edge_keys:
        if key != key_to_save:
            multidigraph.remove_edge(node1, node2, key)


def add_projected_points_as_nodes(
    projected_points: pd.DataFrame, graph: nx.MultiDiGraph
) -> (list[tuple], nx.MultiDiGraph):
    new_graph = graph.copy()
    route_nodes = []
    for _, row in projected_points.iterrows():
        label, start, end, key = row["edge"]
        new_points = row["geometry"].copy()
        postfix = 1

        _remove_multiedges_except_key(new_graph, start, end, key)

        current_edge = new_graph.get_edge_data(start, end)[key]["geometry"]
        current_node = graph.nodes[start]
        current_node = Point(current_node["x"], current_node["y"])
        length = len(new_points)
        new_node_name = None
        while len(new_points) > 0:
            new_node = min(new_points, key=current_node.distance)

            new_node_name = label + (("_" + str(postfix)) if length > 1 else "")
            route_nodes.append((new_node_name, start, end))
            new_graph.add_node(new_node_name, x=new_node.x, y=new_node.y, desc="bus_stop")

            first, _, second = split(current_edge, new_node.buffer(0.0001)).geoms

            new_edge = LineString(list(first.coords[:-1]) + list(new_node.coords))
            new_graph.add_edge(start, new_node_name, length_meter=new_edge.length, geometry=new_edge)

            start = new_node_name
            current_edge = LineString(list(new_node.coords) + list(second.coords[1:]))
            current_node = new_node
            new_points.remove(new_node)
            postfix = postfix + 1

        new_graph.add_edge(new_node_name, end, length_meter=current_edge.length, geometry=current_edge)
    return route_nodes, new_graph


def convert_multidigraph_to_digraph(multidigraph: nx.MultiDiGraph, weight: str = "length_meter") -> nx.DiGraph:
    digraph = nx.DiGraph()

    for node, data in multidigraph.nodes(data=True):
        digraph.add_node(node, **data)

    for u, v, data in multidigraph.edges(data=True):
        if digraph.has_edge(u, v):
            if digraph[u][v][weight] > data[weight]:
                digraph.add_edge(u, v, **data)
        else:
            digraph.add_edge(u, v, **data)
    return digraph


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
            node1 = route_[i]
            node2 = route_[i + 1]
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
