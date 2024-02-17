import geopandas as gpd
import networkx as nx
import pandas as pd

from .base_models import BuildsAvailabilitier, BuildsGrapher, BuildsMatrixer


class Invoker:  # TODO rename to future lib name??
    def __init__(self, city_osm_id: int, city_crs: int):
        self.city_crs = city_crs
        self.city_osm_id = city_osm_id
        # self.graphs: dict[GraphType, nx.Graph] = {} # TODO: add graphs to Invoker

        self._intermodal_graph: nx.MultiDiGraph = None

    # TODO: replace non-alternative OSM integration with optional graphs build from OSM
    # def try_build_graph_from_osm(self, graph_type: GraphType, osm_id: int) -> nx.Graph:
    #     """Build a graph from OSM data and save it as the given graph type"""
    #     raise NotImplementedError()

    # TODO: getter for graphs
    # def get_graph(self, graph_type: GraphType) -> nx.Graph | None:
    #     """Return the graph of given type"""
    #     return self.graphs.get(graph_type)

    # TODO setter for graphs + validation
    # def set_graph(self, graph_type: GraphType, graph: nx.Graph) -> None:
    #     self.graphs[graph_type] = graph
    #     raise NotImplementedError()

    # TODO: update BuildsGrapher logic to construct from the graphs, fail if not all graphs are available
    def get_intermodal_graph(self) -> nx.MultiDiGraph:
        # if not all(graph_type in self.graphs for graph_type in GraphType):
        #     raise ValueError("Some graph types are missing")
        self._intermodal_graph = BuildsGrapher(
            city_osm_id=self.city_osm_id,
            city_crs=self.city_crs,
        ).get_intermodal_graph()
        return self._intermodal_graph

    def get_adjacency_matrix(
        self, buildings_from: gpd.GeoDataFrame, services_to: gpd.GeoDataFrame, weight: str
    ) -> pd.DataFrame:

        if self._intermodal_graph is None:
            self.get_intermodal_graph()

        return BuildsMatrixer(
            buildings_from=buildings_from,
            services_to=services_to,
            weight=weight,
            city_crs=self.city_crs,
            nx_intermodal_graph=self._intermodal_graph,
        ).get_adjacency_matrix()

    def get_accessibility_isochrone(self):
        if self._intermodal_graph is None:
            self.get_intermodal_graph()
        return BuildsAvailabilitier(city_crs=self.city_crs).get_accessibility_isochrone()
