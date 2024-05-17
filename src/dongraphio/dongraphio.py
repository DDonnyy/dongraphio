from typing import Optional, Tuple

import geopandas as gpd
import networkx as nx
import pandas as pd
from loguru import logger

from .base_models import BuildsAvailabilitier, BuildsGrapher, BuildsMatrixer
from .enums import GraphType


class DonGraphio:
    def __init__(self, city_crs: int, intermodal_graph: nx.MultiDiGraph | None = None):
        self.city_crs = city_crs
        # self.graphs: dict[GraphType, nx.Graph] = {} # TODO: add graphs to Invoker

        self._intermodal_graph = intermodal_graph

    # TODO: replace non-alternative OSM integration with optional graphs build from OSM
    # def try_build_graph_from_osm(self, graph_type: GraphType, osm_id: int) -> nx.Graph:
    #     """Build a graph from OSM data and save it as the given graph type"""
    #     raise NotImplementedError()

    def get_graph(self) -> Optional[nx.DiGraph]:
        """
        Return the intermodal graph.

        Returns:
            Optional[nx.DiGraph]: The intermodal graph if it exists, else None.
        Raises:
            RuntimeError: If no graph has been set, call get_intermodal_graph_from_osm() or set it by set_graph().
        """
        if self._intermodal_graph is None:
            raise RuntimeError("No graph has set, call get_intermodal_graph_from_osm() or set it by set_graph()")
        return self._intermodal_graph

    def set_graph(self, graph: nx.DiGraph) -> None:
        """
        Set the intermodal graph for the object.

        Args:
            graph (nx.DiGraph): The graph to be set.
        Returns:
            None
        """
        self._intermodal_graph = graph

    # TODO: update BuildsGrapher logic to construct from the graphs, fail if not all graphs are available

    def get_intermodal_graph_from_osm(self, city_osm_id: int, keep_city_boundary: bool = True) -> nx.MultiDiGraph:
        """
        Retrieves the intermodal graph for a given city from OpenStreetMap.
        Args:
            city_osm_id (int): The OpenStreetMap ID of the city.
            keep_city_boundary (bool, optional): Whether to keep the city boundary in the graph. Defaults to True.
        Returns:
            nx.MultiDiGraph: The intermodal graph representing the city.
        """
        # if not all(graph_type in self.graphs for graph_type in GraphType):
        #     raise ValueError("Some graph types are missing")
        logger.info("Creating intermodal graph from OSM...")
        self._intermodal_graph = BuildsGrapher(
            city_osm_id=city_osm_id, city_crs=self.city_crs, keep_city_boundary=keep_city_boundary
        ).get_intermodal_graph()
        return self._intermodal_graph

    def get_adjacency_matrix(
        self,
        gdf_from: gpd.GeoDataFrame,
        gdf_to: gpd.GeoDataFrame,
        weight: str,
        graph_type: list[GraphType] | None = None,
    ) -> Optional[pd.DataFrame]:
        """
        Calculate the adjacency matrix between the given GeoDataFrames based on
        the specified weight and intermodal graph.

        Args:
            gdf_from (gpd.GeoDataFrame): The GeoDataFrame containing the buildings.
            gdf_to (gpd.GeoDataFrame): The GeoDataFrame containing the services.
            weight (str): The weight attribute, could be only "time_min" or"length_meter".
            graph_type (list[GraphType]): The List of Enum types of the graph to search shortest way.
        Returns:
            Optional[pd.DataFrame]: The adjacency matrix as a DataFrame, or None if the intermodal graph is not set.
        Raises:
            RuntimeError: If no graph has been set, call get_intermodal_graph_from_osm() or set it by set_graph().
        """
        if self._intermodal_graph is None:
            raise RuntimeError("No graph has set, call get_intermodal_graph_from_osm() or set it by set_graph()")
        logger.info("Creating adjacency matrix based on provided graph...")
        to_return = BuildsMatrixer(
            gdf_from=gdf_from,
            gdf_to=gdf_to,
            weight=weight,
            city_crs=self.city_crs,
            nx_intermodal_graph=self._intermodal_graph,
            graph_type=graph_type,
        ).get_adjacency_matrix()
        logger.info("Adjacency matrix done!")
        return to_return

    def get_accessibility_isochrones(
        self, graph_type: list[GraphType], x_from: float, y_from: float, weight_value: int, weight_type: str
    ) -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        """
        Get accessibility isochrones and return three GeoDataFrame objects with isochrones, and
        if graph_type contains GraphType.PUBLIC_TRANSPORT enum - routes and public transport stops.

        Args:
            graph_type (list[GraphType]): The List of Enum types of the graph to build isochrones.
            x_from (float): The x-coordinate of the starting point in the corresponding coordinate system.
            y_from (float): The y-coordinate of the starting point in the corresponding coordinate system.
            weight_value (int): The value of the weight.
            weight_type (str): The type of the weight, could be only "time_min" or "length_meter" .

        Returns:
            (gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]): Isochrones,routes,stops.
        Raises:
            RuntimeError: If no graph has been set, call get_intermodal_graph_from_osm() or set it by set_graph().
        """
        # Check if intermodal graph is set
        if self._intermodal_graph is None:
            raise RuntimeError("No graph has set, call get_intermodal_graph_from_osm() or set it by set_graph()")
        # Build accessibility isochrones
        logger.info("Creating accessibility isochrones based on provided graph and point...")
        to_return = BuildsAvailabilitier(
            graph_type=graph_type,
            city_crs=self.city_crs,
            x_from=x_from,
            y_from=y_from,
            weight_value=weight_value,
            weight_type=weight_type,
            nx_intermodal_graph=self._intermodal_graph,
        ).get_accessibility_isochrone()
        logger.info("Accessibility isochrones done!\n")
        return to_return
