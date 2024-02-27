import json
import logging
from typing import Literal

import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from pydantic import BaseModel, InstanceOf

from .enums import GraphType
from .utils import graphs, matrix_utils


class BuildsGrapher(BaseModel):
    city_osm_id: int
    city_crs: int
    public_transport_speeds: dict = None
    walk_speed: dict = None
    drive_speed: dict = None
    gdf_files: dict = None

    def get_intermodal_graph(self) -> nx.MultiDiGraph:
        G_public_transport: nx.MultiDiGraph = graphs.get_public_trasport_graph(
            self.city_osm_id, self.city_crs, self.gdf_files, self.public_transport_speeds
        )
        G_walk: nx.MultiDiGraph = graphs.get_osmnx_graph(self.city_osm_id, self.city_crs, "walk", speed=self.walk_speed)

        G_drive: nx.MultiDiGraph = graphs.get_osmnx_graph(
            self.city_osm_id, self.city_crs, "drive", speed=self.drive_speed
        )
        logging.info("Union of city_graphs...")
        nx_intermodal_graph: nx.MultiDiGraph = graphs.graphs_spatial_union(G_walk, G_drive)
        if G_public_transport.number_of_edges() > 0:
            nx_intermodal_graph: nx.MultiDiGraph = graphs.graphs_spatial_union(nx_intermodal_graph, G_public_transport)

        for _u, _v, d in nx_intermodal_graph.edges(data=True):
            if "time_min" not in d:
                d["time_min"] = round(d["length_meter"] / G_walk.graph["walk speed"], 2)
            if "desc" not in d:
                d["desc"] = ""

        for _u, d in nx_intermodal_graph.nodes(data=True):
            if "stop" not in d:
                d["stop"] = "False"
            if "desc" not in d:
                d["desc"] = ""

        nx_intermodal_graph.graph["graph_type"] = "intermodal graph"
        nx_intermodal_graph.graph["car speed"] = G_drive.graph["car speed"]
        nx_intermodal_graph.graph.update({k: v for k, v in G_public_transport.graph.items() if "speed" in k})
        logging.info("Intermodal graph done!")
        return nx_intermodal_graph


class BuildsMatrixer(BaseModel):
    buildings_from: InstanceOf[gpd.GeoDataFrame]
    services_to: InstanceOf[gpd.GeoDataFrame]
    weight: Literal["time_min", "length_meter"]
    city_crs: int
    nx_intermodal_graph: InstanceOf[nx.DiGraph]

    def get_adjacency_matrix(self) -> pd.DataFrame:
        buildings_from = self.buildings_from.to_crs(self.city_crs)
        services_to = self.services_to.to_crs(self.city_crs)
        mobility_sub_graph = matrix_utils.get_subgraph(
            self.nx_intermodal_graph.copy(),
            "type",
            [
                t.value
                for t in set().union(*(d.value.get("types") for d in (GraphType.PUBLIC_TRANSPORT, GraphType.WALK)))
            ],
        )

        nk_graph = matrix_utils.convert_nx2nk(
            mobility_sub_graph, idmap=matrix_utils.get_nx2nk_idmap(mobility_sub_graph), weight=self.weight
        )

        graph_with_geom = matrix_utils.load_graph_geometry(self.nx_intermodal_graph)
        df = pd.DataFrame.from_dict(dict(graph_with_geom.nodes(data=True)), orient="index")
        graph_gdf = gpd.GeoDataFrame(df, geometry=df["geometry"], crs=self.city_crs)
        from_houses = graph_gdf["geometry"].sindex.nearest(
            buildings_from["geometry"], return_distance=True, return_all=False
        )
        to_services = graph_gdf["geometry"].sindex.nearest(
            services_to["geometry"], return_distance=True, return_all=False
        )
        distance_matrix = pd.DataFrame(0, index=to_services[0][1], columns=from_houses[0][1])

        splited_matrix = np.array_split(
            distance_matrix.copy(deep=True),
            int(len(distance_matrix) / 1000) + 1,
        )

        # TODO use a* instead of dijkstra
        for i in range(len(splited_matrix)):  # pylint: disable=consider-using-enumerate
            r = nk.distance.SPSP(G=nk_graph, sources=splited_matrix[i].index.values).run()
            splited_matrix[i] = splited_matrix[i].apply(lambda x: matrix_utils.get_nk_distances(r, x), axis=1)
            del r
        distance_matrix: pd.DataFrame = pd.concat(splited_matrix)
        del splited_matrix
        distance_matrix.index = services_to.index
        distance_matrix.columns = buildings_from.index
        return distance_matrix


class BuildsAvailabilitier(BaseModel):
    graph_type: list[GraphType]
    city_crs: int
    x_from: list  # TODO add single int to list validator
    y_from: list  # TODO add single int to list validator
    weight_value: int
    weight_type: Literal["time_min", "length_meter"]
    nx_intermodal_graph: InstanceOf[nx.DiGraph]
    _edge_types = None

    def get_accessibility_isochrone(self):
        def _get_nk_distances(nk_dists, loc):
            target_nodes = loc.index.astype("int")
            source_node = loc.name
            distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]
            return pd.Series(data=distances, index=target_nodes)

        source = pd.DataFrame(
            data=list(zip(range(len(self.x_from)), self.x_from, self.y_from)), columns=["id", "x", "y"]
        )

        source = gpd.GeoDataFrame(source, geometry=gpd.points_from_xy(source["x"], source["y"], crs=self.city_crs))
        self._edge_types = [t.value for t in set().union(*(d.value.get("types") for d in self.graph_type))]
        mobility_graph = matrix_utils.get_subgraph(self.nx_intermodal_graph, "type", self._edge_types)

        mobility_graph = nx.convert_node_labels_to_integers(mobility_graph)
        graph_df = pd.DataFrame.from_dict(dict(mobility_graph.nodes(data=True)), orient="index")
        graph_gdf = gpd.GeoDataFrame(graph_df, geometry=gpd.points_from_xy(graph_df["x"], graph_df["y"])).set_crs(
            self.city_crs
        )

        from_sources = graph_gdf["geometry"].sindex.nearest(source["geometry"], return_distance=True, return_all=False)

        distances = pd.DataFrame(0, index=from_sources[0][1], columns=list(mobility_graph.nodes()))

        nk_graph = matrix_utils.convert_nx2nk(
            mobility_graph, idmap=matrix_utils.get_nx2nk_idmap(mobility_graph), weight=self.weight_type
        )
        logging.info("Before finding dis")
        nk_dists = nk.distance.SPSP(G=nk_graph, sources=distances.index.values).run()
        logging.info("After finding dis")
        distances = distances.apply(lambda x: _get_nk_distances(nk_dists, x), axis=1)

        dist_nearest = pd.DataFrame(data=from_sources[1], index=from_sources[0][1], columns=["dist"])
        walk_speed = 4 * 1000 / 60
        dist_nearest = dist_nearest / walk_speed if self.weight_type == "time_min" else dist_nearest
        distances = distances.add(dist_nearest.dist, axis=0)

        cols = distances.columns.to_numpy()
        source["isochrone_nodes"] = [cols[x].tolist() for x in distances.le(self.weight_value).to_numpy()]

        for x, y in list(zip(from_sources[0][1], source["isochrone_nodes"])):
            y.extend([x])

        source["isochrone_geometry"] = source["isochrone_nodes"].apply(
            lambda x: [graph_gdf["geometry"].loc[[y for y in x]]]
        )
        source["isochrone_geometry"] = [list(x[0].geometry) for x in source["isochrone_geometry"]]
        source["isochrone_geometry"] = [[y.buffer(0.01) for y in x] for x in source["isochrone_geometry"]]
        source["isochrone_geometry"] = source["isochrone_geometry"].apply(
            lambda x: shapely.ops.cascaded_union(x).convex_hull
        )
        source["isochrone_geometry"] = gpd.GeoSeries(source["isochrone_geometry"], crs=self.city_crs)

        isochrones = [
            gpd.GeoDataFrame(
                {
                    "travel_type": [str([d.value.get("name") for d in self.graph_type])],
                    "weight_type": [self.weight_type],
                    "weight_value": [self.weight_value],
                    "geometry": [x],
                },
                geometry=[x],
                crs=self.city_crs,
            )  # .to_crs(4326)
            for x in source["isochrone_geometry"]
        ]
        stops, routes = ([None], [None])
        isochrones = pd.concat([x for x in isochrones])
        if GraphType.PUBLIC_TRANSPORT in self.graph_type and self.weight_type == "time_min":
            stops, routes = self._get_routes(graph_gdf, source["isochrone_nodes"], mobility_graph)
        return isochrones, routes, stops

    def _get_routes(self, graph_gdf, selected_nodes, mobility_graph):
        stops = graph_gdf[graph_gdf["stop"] == "True"]
        stop_types = stops["desc"].apply(lambda x: pd.Series({t: True for t in x.split(", ")}), type).fillna(False)
        stops = stops.join(stop_types)
        stops.reset_index(inplace=True)
        stops.rename(columns={"index": "nodeID"}, inplace=True)
        stops_result = [stops.loc[stops["nodeID"].isin(x)].to_crs(4326) for x in selected_nodes]
        nodes = [x["nodeID"] for x in stops_result]
        subgraph = [mobility_graph.subgraph(x) for x in nodes]
        routes = [pd.DataFrame.from_records([e[-1] for e in x.edges(data=True, keys=True)]) for x in subgraph]

        def routes_selection(routes):
            if len(routes) > 0:
                routes_select = routes[routes["type"].isin(self._edge_types[:-1])]
                routes_select["geometry"] = routes_select["geometry"].apply(lambda x: shapely.wkt.loads(x))
                routes_select = routes_select[["type", "time_min", "length_meter", "geometry"]]
                routes_select = gpd.GeoDataFrame(routes_select, crs=self.city_crs).to_crs(4326)
                return routes_select
            else:
                return None

        routes_result = list(map(routes_selection, routes))
        routes_result = gpd.GeoDataFrame(data=routes_result[0],geometry="geometry")
        stops_result = gpd.GeoDataFrame(data = stops_result[0],geometry="geometry")
        return stops_result, routes_result
