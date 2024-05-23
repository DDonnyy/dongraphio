from typing import Literal, Optional, Tuple

import geopandas as gpd
import networkit as nk
import networkx as nx
import pandas as pd
from loguru import logger
from pydantic import BaseModel, InstanceOf, field_validator
from shapely import Point, from_wkt
from shapely.ops import unary_union
from tqdm.auto import tqdm

from ..enums import GraphType
from ..utils import convert_multidigraph_to_digraph, get_nx2nk_idmap, matrix_utils, nx_to_gdf

tqdm.pandas()


class BuildsAvailabilitier(BaseModel):
    graph_type: list[GraphType]
    city_crs: int
    points: InstanceOf[gpd.GeoSeries] | InstanceOf[Point]
    weight_value: int
    weight_type: Literal["time_min", "length_meter"]
    nx_intermodal_graph: InstanceOf[nx.DiGraph]
    _edge_types = None

    @field_validator("points")
    @classmethod
    def ensure_points(cls, points):
        if isinstance(points, gpd.GeoSeries):
            assert points.apply(
                lambda geom: isinstance(geom, Point)
            ).all(), "Geometry type in provided points is not Point"
        else:
            points = gpd.GeoSeries([points])
        return points

    def get_accessibility_isochrone(
        self,
    ) -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:

        self._edge_types = [d.value for d in sum([t.edges for t in self.graph_type], [])]

        mobility_graph = matrix_utils.get_subgraph(self.nx_intermodal_graph, "type", self._edge_types)
        mobility_graph = nx.convert_node_labels_to_integers(mobility_graph)

        if mobility_graph.is_multigraph() & mobility_graph.is_directed():
            mobility_graph = convert_multidigraph_to_digraph(mobility_graph, self.weight_type)
        mobility_graph.graph["crs"] = self.city_crs
        graph_gdf = nx_to_gdf(nx.MultiDiGraph(mobility_graph), nodes=True)

        from_sources = graph_gdf["geometry"].sindex.nearest(self.points, return_distance=True, return_all=False)
        source_index = from_sources[0][1]
        distances = pd.DataFrame(float(0), index=source_index, columns=list(mobility_graph.nodes()))

        logger.debug("Calculating distances from the specified point...")

        mapping = get_nx2nk_idmap(mobility_graph)
        nk_graph = nk.nxadapter.nx2nk(mobility_graph, self.weight_type)
        spsp = nk.distance.SPSP(G=nk_graph, sources=distances.index.values).run()
        for index, _ in distances.iterrows():
            for column in distances.columns:
                distances.loc[index, column] = spsp.getDistance(mapping.get(index), mapping.get(column))
        del spsp

        dist_nearest = pd.DataFrame(data=from_sources[1], index=from_sources[0][1], columns=["dist"])
        walk_speed = 4 * 1000 / 60
        dist_nearest = dist_nearest / walk_speed if self.weight_type == "time_min" else dist_nearest
        distances = distances.add(dist_nearest.dist, axis=0).transpose()

        distances = distances[distances[source_index] <= self.weight_value]
        results = []
        point_num = 0
        logger.debug("Building isochrones geometry...")
        for column_name in distances.columns:
            geometry = []
            for ind in distances.index:
                value = distances.loc[ind, column_name]
                if not pd.isna(value):
                    geometry.append(
                        graph_gdf.loc[ind].geometry.buffer(round(self.weight_value - value, 2) * walk_speed * 0.8)
                        if self.weight_type == "time_min"
                        else graph_gdf.loc[ind].geometry.buffer(round(self.weight_value - value, 2) * 0.8)
                    )
            geometry = unary_union(geometry)
            results.append({"geometry": geometry, "point": str(self.points.iloc[point_num])})
            point_num += 1

        isochrones = gpd.GeoDataFrame(data=results, geometry="geometry", crs=self.city_crs)
        isochrones["travel_type"] = str([d.russian_name for d in self.graph_type])
        isochrones["weight_type"] = self.weight_type
        isochrones["weight_value"] = self.weight_value

        stops, routes = (None, None)
        if GraphType.PUBLIC_TRANSPORT in self.graph_type and self.weight_type == "time_min":
            if not (graph_gdf[graph_gdf["stop"] == "True"]).empty:
                stops, routes = self._get_routes(graph_gdf, distances.index.values, mobility_graph)
            else:
                logger.info("No public transport node in graph")
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
        selected_nodes = [node for node in selected_nodes if node in stops.index]
        subgraph = mobility_graph.subgraph(selected_nodes)
        routes = pd.DataFrame.from_records([e[-1] for e in subgraph.edges(data=True)])
        routes["geometry"] = routes["geometry"].apply(from_wkt)
        routes_result = gpd.GeoDataFrame(data=routes, geometry="geometry", crs=graph_gdf.crs)
        stops_result = gpd.GeoDataFrame(data=stops.loc[selected_nodes], geometry="geometry", crs=graph_gdf.crs)
        return stops_result, routes_result
