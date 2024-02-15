import geopandas as gpd
import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field

from .base_models import BuildsGrapher, BuildsMatrixer, BuildsAvailabilitier


class Invoker(BaseModel):  # TODO rename to future lib name??
    city_osm_id: int = Field(gt=0)
    city_crs: int = Field(gt=0)
    _intermodal_graph: nx.MultiDiGraph = None

    def get_intermodal_graph(self) -> nx.MultiDiGraph:
        self._intermodal_graph = BuildsGrapher(city_osm_id=self.city_osm_id,
                                               city_crs=self.city_crs).get_intermodal_graph()
        return self._intermodal_graph

    def get_adjacency_matrix(self, buildings_from: gpd.GeoDataFrame, services_to: gpd.GeoDataFrame,
                             weight: str) -> pd.DataFrame:

        if self._intermodal_graph is None:
            self.get_intermodal_graph()

        return BuildsMatrixer(buildings_from=buildings_from, services_to=services_to, weight=weight,
                              city_crs=self.city_crs, nx_intermodal_graph=self._intermodal_graph).get_adjacency_matrix()

    def get_accessibility_isochrone(self):
        if self._intermodal_graph is None:
            self.get_intermodal_graph()
        return BuildsAvailabilitier(city_crs=self.city_crs).get_accessibility_isochrone()
