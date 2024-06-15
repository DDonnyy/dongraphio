from typing import Literal, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, InstanceOf, field_validator
from scipy.spatial import KDTree
from shapely import Point, from_wkt
from shapely.ops import unary_union
from tqdm.auto import tqdm

from ..enums import GraphType
from ..utils import matrix_utils

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

        nodes_with_data = list(mobility_graph.nodes(data=True))
        logger.info("Calculating isochrones distances...")
        coordinates = np.array([(data["x"], data["y"]) for node, data in nodes_with_data])
        tree = KDTree(coordinates)
        target_coord = [(p.x, p.y) for p in self.points]
        distance, indices = tree.query(target_coord)
        nearest_nodes = [nodes_with_data[index][0] for index in indices]

        # TODO do not raise if duplicates
        if len(set(nearest_nodes)) != len(nearest_nodes):
            raise RuntimeError("Some points are too close to each other,check projection/geometry")

        dist_nearest = pd.DataFrame(data=distance, index=nearest_nodes, columns=["dist"])
        walk_speed = 4 * 1000 / 60
        dist_nearest = dist_nearest / walk_speed if self.weight_type == "time_min" else dist_nearest

        if (dist_nearest > self.weight_value).all().all():
            raise RuntimeError(
                "The point(s) lie further from the graph than weight_value, it's impossible to "
                "construct isochrones. Check the coordinates of the point(s)/their projection"
            )

        data = []
        for source in nearest_nodes:
            dist, path = nx.single_source_dijkstra(
                mobility_graph, source, weight=self.weight_type, cutoff=self.weight_value
            )
            for node_from, way in path.items():
                source = way[0]
                destination = node_from
                distance = dist.get(node_from, np.nan)
                data.append((source, destination, distance))

        dist_matrix = pd.DataFrame(data, columns=["source", "destination", "distance"])
        dist_matrix = dist_matrix.pivot(index="source", columns="destination", values="distance")
        dist_matrix = dist_matrix.add(dist_nearest.dist, axis=0).transpose()
        dist_matrix = dist_matrix.mask(dist_matrix >= self.weight_value, np.nan)
        dist_matrix.dropna(how="all", inplace=True)

        results = []
        point_num = 0
        logger.info("Building isochrones geometry...")

        for column_name in dist_matrix.columns:
            geometry = []
            for ind in dist_matrix.index:
                value = dist_matrix.loc[ind, column_name]
                if not pd.isna(value):
                    node = mobility_graph.nodes[ind]
                    point = Point(node["x"], node["y"])
                    geometry.append(
                        point.buffer(round(self.weight_value - value, 2) * walk_speed * 0.8)
                        if self.weight_type == "time_min"
                        else point.buffer(round(self.weight_value - value, 2) * 0.8)
                    )
            geometry = unary_union(geometry)
            results.append({"geometry": geometry, "point": str(self.points.iloc[point_num]), "point_number": point_num})
            point_num += 1

        isochrones = gpd.GeoDataFrame(data=results, geometry="geometry", crs=self.city_crs)
        isochrones["travel_type"] = str([d.russian_name for d in self.graph_type])
        isochrones["weight_type"] = self.weight_type
        isochrones["weight_value"] = self.weight_value
        stops, routes = (None, None)
        if GraphType.PUBLIC_TRANSPORT in self.graph_type and self.weight_type == "time_min":
            node_data = {
                node: mobility_graph.nodes[node]
                for node in dist_matrix.index
                if mobility_graph.nodes[node]["stop"] == "True"
            }
            if len(node_data) > 0:
                logger.info("Building public transport geometry...")
                stops, routes = self._get_routes(node_data, mobility_graph)
            else:
                logger.info("No public transport node in graph")
        return isochrones, routes, stops

    def _get_routes(self, stops, mobility_graph):
        stops = pd.DataFrame(stops).T
        stops["geometry"] = stops.apply(lambda x: Point(x.x, x.y), axis=1)
        stops.drop(columns=["x", "y"], inplace=True)

        subgraph = mobility_graph.subgraph(stops.index)
        routes = pd.DataFrame.from_records([e[-1] for e in subgraph.edges(data=True)])
        if routes.empty:
            return None, None
        routes["geometry"] = routes["geometry"].apply(lambda x: from_wkt(str(x)))
        routes_result = gpd.GeoDataFrame(data=routes, geometry="geometry", crs=self.city_crs)
        stops_result = gpd.GeoDataFrame(data=stops, geometry="geometry", crs=self.city_crs)
        return stops_result, routes_result
