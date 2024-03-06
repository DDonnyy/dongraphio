from typing import Literal, Optional, Tuple

import geopandas as gpd
import networkit as nk
import networkx as nx
import pandas as pd
import shapely
from loguru import logger
from pydantic import BaseModel, InstanceOf
from tqdm.auto import tqdm

from .enums import GraphType
from .utils import graphs, matrix_utils

tqdm.pandas()


class BuildsGrapher(BaseModel):
    city_osm_id: int
    city_crs: int
    keep_city_boundary: bool = True
    public_transport_speeds: dict = None
    walk_speed: dict = None
    drive_speed: dict = None
    gdf_files: dict = None

    def get_intermodal_graph(self) -> nx.MultiDiGraph:
        G_public_transport: nx.MultiDiGraph = graphs.get_public_trasport_graph(
            self.city_osm_id, self.city_crs, self.gdf_files, self.keep_city_boundary, self.public_transport_speeds
        )
        G_walk: nx.MultiDiGraph = graphs.get_osmnx_graph(self.city_osm_id, self.city_crs, "walk", speed=self.walk_speed)

        G_drive: nx.MultiDiGraph = graphs.get_osmnx_graph(
            self.city_osm_id, self.city_crs, "drive", speed=self.drive_speed
        )
        logger.debug("Preparing union of city_graphs...")
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
        logger.info("Intermodal graph done!\n")
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
            [t.value for t in (GraphType.PUBLIC_TRANSPORT.edges + GraphType.WALK.edges)],
        )

        nk_graph = matrix_utils.convert_nx2nk(
            mobility_sub_graph, idmap=matrix_utils.get_nx2nk_idmap(mobility_sub_graph), weight=self.weight
        )

        graph_with_geom = matrix_utils.load_graph_geometry(mobility_sub_graph)
        df = pd.DataFrame.from_dict(dict(graph_with_geom.nodes(data=True)), orient="index")
        graph_gdf = gpd.GeoDataFrame(df, geometry=df["geometry"], crs=self.city_crs)
        from_houses = graph_gdf["geometry"].sindex.nearest(
            buildings_from["geometry"], return_distance=True, return_all=False
        )
        to_services = graph_gdf["geometry"].sindex.nearest(
            services_to["geometry"], return_distance=True, return_all=False
        )
        distance_matrix = pd.DataFrame(0, index=to_services[0][1], columns=from_houses[0][1])

        # TODO use a* instead of dijkstra?
        logger.debug("Calculating distances from buildings to services ...")
        r = nk.distance.SPSP(G=nk_graph, sources=distance_matrix.index.values).run()
        distance_matrix = distance_matrix.apply(lambda x: matrix_utils.get_nk_distances(r, x), axis=1)
        del r
        distance_matrix.index = services_to.index
        distance_matrix.columns = buildings_from.index
        return distance_matrix


class BuildsAvailabilitier(BaseModel):
    graph_type: list[GraphType]
    city_crs: int
    x_from: float
    y_from: float
    weight_value: int
    weight_type: Literal["time_min", "length_meter"]
    nx_intermodal_graph: InstanceOf[nx.DiGraph]
    _edge_types = None

    def get_accessibility_isochrone(
        self,
    ) -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        def _get_nk_distances(nk_dists, loc):
            target_nodes = loc.index.astype("int")
            source_node = loc.name
            distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]
            return pd.Series(data=distances, index=target_nodes)

        source = gpd.GeoDataFrame(geometry=gpd.points_from_xy([self.x_from], [self.y_from], crs=self.city_crs))

        self._edge_types = [d.value for d in sum([t.edges for t in self.graph_type], [])]

        mobility_graph = matrix_utils.get_subgraph(self.nx_intermodal_graph, "type", self._edge_types)
        mobility_graph = nx.convert_node_labels_to_integers(mobility_graph)
        graph_df = pd.DataFrame.from_dict(dict(mobility_graph.nodes(data=True)), orient="index")
        graph_gdf = gpd.GeoDataFrame(graph_df, geometry=gpd.points_from_xy(graph_df["x"], graph_df["y"])).set_crs(
            self.city_crs
        )

        from_sources = graph_gdf["geometry"].sindex.nearest(source["geometry"], return_distance=True, return_all=False)
        source_index = from_sources[0][1]
        distances = pd.DataFrame(0, index=source_index, columns=list(mobility_graph.nodes()))

        nk_graph = matrix_utils.convert_nx2nk(
            mobility_graph, idmap=matrix_utils.get_nx2nk_idmap(mobility_graph), weight=self.weight_type
        )

        nk_dists = nk.distance.SPSP(G=nk_graph, sources=distances.index.values).run()
        logger.debug("Calculating distances from the specified point...")
        distances = distances.apply(lambda x: _get_nk_distances(nk_dists, x), axis=1)

        dist_nearest = pd.DataFrame(data=from_sources[1], index=from_sources[0][1], columns=["dist"])
        walk_speed = 4 * 1000 / 60
        dist_nearest = dist_nearest / walk_speed if self.weight_type == "time_min" else dist_nearest
        distances = distances.add(dist_nearest.dist, axis=0).transpose()

        distances = distances[distances[source_index[0]] <= self.weight_value]

        logger.debug("Building isochrones geometry...")
        distances["geometry"] = distances.apply(
            lambda x: (
                graph_gdf.loc[x.index].geometry.buffer(round(self.weight_value - x, 2) * walk_speed * 0.8)
                if self.weight_type == "time_min"
                else graph_gdf.loc[x.index].geometry.buffer(round(self.weight_value - x, 2) * 0.8)
            )
        )
        isochrone_geometry = gpd.GeoDataFrame(data=distances, geometry="geometry").geometry.unary_union

        isochrones = gpd.GeoDataFrame(
            {
                "travel_type": [str([d.russian_name for d in self.graph_type])],
                "weight_type": [self.weight_type],
                "weight_value": [self.weight_value],
                "geometry": [isochrone_geometry],
            },
            geometry="geometry",
            crs=self.city_crs,
        )

        stops, routes = (None, None)

        if GraphType.PUBLIC_TRANSPORT in self.graph_type and self.weight_type == "time_min":
            stops, routes = self._get_routes(graph_gdf, distances.index.values, mobility_graph)
        return isochrones, routes, stops

    def _get_routes(self, graph_gdf, selected_nodes, mobility_graph):
        stops = graph_gdf[graph_gdf["stop"] == "True"]
        stop_types = (
            stops["desc"]
            .astype(object)
            .apply(lambda x: pd.Series({t: True for t in x.split(", ")}))
            .astype(bool)
            .fillna(False)
            .infer_objects(copy=False)
        )
        stops = stops.join(stop_types)
        stops.reset_index(inplace=True)
        stops.rename(columns={"index": "nodeID"}, inplace=True)
        stops_result = [stops.loc[stops["nodeID"].isin(selected_nodes)]]
        nodes = [x["nodeID"] for x in stops_result]
        subgraph = [mobility_graph.subgraph(x) for x in nodes]
        routes = [pd.DataFrame.from_records([e[-1] for e in x.edges(data=True, keys=True)]) for x in subgraph]

        def routes_selection(routes):
            if len(routes) > 0:
                routes_select = routes[routes["type"].isin(self._edge_types)]
                routes_select["geometry"] = routes_select["geometry"].apply(lambda x: shapely.wkt.loads(x))
                routes_select = routes_select[["type", "time_min", "length_meter", "geometry"]]
                routes_select = gpd.GeoDataFrame(routes_select, crs=self.city_crs)
                return routes_select
            return None

        routes_result = list(map(routes_selection, routes))
        routes_result = gpd.GeoDataFrame(data=routes_result[0], geometry="geometry")
        stops_result = gpd.GeoDataFrame(data=stops_result[0], geometry="geometry")
        return stops_result, routes_result
