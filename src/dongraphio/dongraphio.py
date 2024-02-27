import geopandas as gpd
import networkx as nx
import pandas as pd
import logging

from pandas import DataFrame

from .base_models import BuildsAvailabilitier, BuildsGrapher, BuildsMatrixer


class DonGraphio:
    def __init__(self, city_osm_id: int, city_crs: int):
        self.city_crs = city_crs
        self.city_osm_id = city_osm_id
        # self.graphs: dict[GraphType, nx.Graph] = {} # TODO: add graphs to Invoker

        self._intermodal_graph: nx.MultiDiGraph = None

    # TODO: replace non-alternative OSM integration with optional graphs build from OSM
    # def try_build_graph_from_osm(self, graph_type: GraphType, osm_id: int) -> nx.Graph:
    #     """Build a graph from OSM data and save it as the given graph type"""
    #     raise NotImplementedError()

    def get_graph(self) -> nx.Graph | None:
        """Return the graph of given type"""
        return self._intermodal_graph

    def set_graph(self, graph: nx.DiGraph) -> None:
        self._intermodal_graph = graph

    # TODO: update BuildsGrapher logic to construct from the graphs, fail if not all graphs are available

    def get_intermodal_graph_from_osm(self) -> nx.MultiDiGraph:
        # if not all(graph_type in self.graphs for graph_type in GraphType):
        #     raise ValueError("Some graph types are missing")
        self._intermodal_graph = BuildsGrapher(
            city_osm_id=self.city_osm_id,
            city_crs=self.city_crs,
        ).get_intermodal_graph()
        return self._intermodal_graph

    def get_adjacency_matrix(
        self, buildings_from: gpd.GeoDataFrame, services_to: gpd.GeoDataFrame, weight: str
    ) -> DataFrame | None:

        if self._intermodal_graph is None:
            logging.info("No graph has set, call get_intermodal_graph_from_osm() or set it by set_graph()")
            return None
        return BuildsMatrixer(
            buildings_from=buildings_from,
            services_to=services_to,
            weight=weight,
            city_crs=self.city_crs,
            nx_intermodal_graph=self._intermodal_graph,
        ).get_adjacency_matrix()

    def get_accessibility_isochrone(self):
        if self._intermodal_graph is None:
            logging.info("No graph has set, call get_intermodal_graph_from_osm() or set it by set_graph()")
            return None
        return BuildsAvailabilitier(city_crs=self.city_crs).get_accessibility_isochrone()
