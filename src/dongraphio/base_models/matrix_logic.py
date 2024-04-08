from typing import Literal, Optional

import geopandas as gpd
import networkx as nx
import pandas as pd
from loguru import logger
from pydantic import BaseModel, InstanceOf
from tqdm.auto import tqdm

from ..enums import GraphType
from ..utils import convert_multidigraph_to_digraph, get_dist_matrix, get_subgraph, nx_to_gdf

pd.options.mode.chained_assignment = None
tqdm.pandas()


class BuildsMatrixer(BaseModel):
    gdf_from: InstanceOf[gpd.GeoDataFrame]
    gdf_to: InstanceOf[gpd.GeoDataFrame]
    weight: Literal["time_min", "length_meter"]
    city_crs: int
    nx_intermodal_graph: InstanceOf[nx.DiGraph]
    graph_type: Optional[list[GraphType]] = [GraphType.WALK, GraphType.PUBLIC_TRANSPORT]

    def get_adjacency_matrix(self) -> pd.DataFrame:
        self.gdf_from.to_crs(self.city_crs, inplace=True)
        self.gdf_to.to_crs(self.city_crs, inplace=True)
        mobility_sub_graph = get_subgraph(
            self.nx_intermodal_graph,
            "type",
            [t.value for t in sum([t.edges for t in self.graph_type], [])],
        )

        if mobility_sub_graph.is_multigraph() & mobility_sub_graph.is_directed():
            mobility_sub_graph = convert_multidigraph_to_digraph(mobility_sub_graph, self.weight)

        mobility_sub_graph.graph["crs"] = self.city_crs

        graph_gdf = nx_to_gdf(nx.MultiDiGraph(mobility_sub_graph), nodes=True)

        from_ = gpd.sjoin_nearest(self.gdf_from, graph_gdf)
        from_ = from_.reset_index().drop_duplicates(subset="index", keep="first").set_index('index')

        to_ = gpd.sjoin_nearest(self.gdf_to, graph_gdf)
        to_ = to_.reset_index().drop_duplicates(subset="index", keep="first").set_index("index")

        from_index = from_["index_right"]
        to_index = to_["index_right"]
        distance_matrix = pd.DataFrame(float(0), index=from_index, columns=to_index)
        from_index = list(set(from_index))
        to_index = list(set(to_index))
        logger.debug("Calculating distances from buildings to services ...")

        distance_matrix_result = get_dist_matrix(
            mobility_sub_graph,
            from_index,
            to_index,
            weight=self.weight,
        )

        distance_matrix_result = distance_matrix_result.apply(pd.to_numeric, errors="coerce")
        def update_value(ind, col):
            if ind in distance_matrix_result.index and col in distance_matrix_result.columns:
                return distance_matrix_result.loc[ind, col]

        distance_matrix = distance_matrix.apply(lambda x: x.index.map(lambda ind: update_value(ind, x.name)))
        distance_matrix.index = self.gdf_from.index
        distance_matrix.columns = self.gdf_to.index
        return distance_matrix
