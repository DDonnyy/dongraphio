from typing import Optional, Tuple

import geopandas as gpd
import networkx as nx
import pandas as pd
from loguru import logger

from .base_models import BuildsAvailabilitier, OSMGrapher, BuildsMatrixer
from .enums import GraphType


def get_intermodal_graph_from_osm(city_osm_id: int, city_crs: int, keep_city_boundary: bool = True) -> nx.MultiDiGraph:
    """
    Retrieves the intermodal graph for a given city from OpenStreetMap.

    Args:
        city_osm_id (int): The OpenStreetMap ID of the city.
        city_crs (int): The Coordinate Reference System (CRS) for the city.
        keep_city_boundary (bool, optional): Indicates whether to keep the city boundary in the graph. Defaults to True.

    Returns:
        nx.MultiDiGraph: The intermodal graph representing the city.
    """

    logger.info("Creating intermodal graph from OSM...")
    return OSMGrapher(
        city_osm_id=city_osm_id, keep_city_boundary=keep_city_boundary, city_crs=city_crs
    ).get_intermodal_graph()


def get_adjacency_matrix(
    gdf_from: gpd.GeoDataFrame,
    gdf_to: gpd.GeoDataFrame,
    weight: str,
    graph: nx.Graph,
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
    logger.info("Creating adjacency matrix based on provided graph...")
    to_return = BuildsMatrixer(
        gdf_from=gdf_from,
        gdf_to=gdf_to,
        weight=weight,
        nx_intermodal_graph=graph,
        graph_type=graph_type,
    ).get_adjacency_matrix()
    logger.info("Adjacency matrix done!")
    return to_return


def get_accessibility_isochrones(
    graph: nx.Graph, graph_type: list[GraphType], x_from: float, y_from: float, weight_value: int, weight_type: str
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

    logger.info("Creating accessibility isochrones based on provided graph and point...")
    to_return = BuildsAvailabilitier(
        graph_type=graph_type,
        x_from=x_from,
        y_from=y_from,
        weight_value=weight_value,
        weight_type=weight_type,
        nx_intermodal_graph=graph,
    ).get_accessibility_isochrone()
    logger.info("Accessibility isochrones done!\n")
    return to_return
