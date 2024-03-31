from typing import Literal, Optional, Tuple

import geopandas as gpd
import networkit as nk
import networkx as nx
import pandas as pd
import shapely
from loguru import logger
from pydantic import BaseModel, InstanceOf
from tqdm.auto import tqdm

from ..enums import GraphType
from ..utils import convert_multidigraph_to_digraph, get_nx2nk_idmap, matrix_utils, nx_to_gdf

tqdm.pandas()


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

        source = gpd.GeoDataFrame(geometry=gpd.points_from_xy([self.x_from], [self.y_from], crs=self.city_crs))

        self._edge_types = [d.value for d in sum([t.edges for t in self.graph_type], [])]

        mobility_graph = matrix_utils.get_subgraph(self.nx_intermodal_graph, "type", self._edge_types)
        mobility_graph = nx.convert_node_labels_to_integers(mobility_graph)

        if mobility_graph.is_multigraph() & mobility_graph.is_directed():
            mobility_graph = convert_multidigraph_to_digraph(mobility_graph, self.weight_type)
        mobility_graph.graph["crs"] = self.city_crs
        graph_gdf = nx_to_gdf(nx.MultiDiGraph(mobility_graph), nodes=True)

        from_sources = graph_gdf["geometry"].sindex.nearest(source["geometry"], return_distance=True, return_all=False)
        source_index = from_sources[0][1]
        distances = pd.DataFrame(float(0), index=source_index, columns=list(mobility_graph.nodes()))

        logger.debug("Calculating distances from the specified point...")

        mapping = get_nx2nk_idmap(mobility_graph)
        nk_graph = nk.nxadapter.nx2nk(mobility_graph, self.weight_type)
        spsp = nk.distance.SPSP(G=nk_graph, sources=distances.index.values).run()
        for index, row in distances.iterrows():
            for column in distances.columns:
                distances.loc[index, column] = spsp.getDistance(mapping.get(index), mapping.get(column))
        del spsp

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
