from typing import Any

import networkx as nx
from geopandas import GeoDataFrame
from networkit import distance
from networkx import MultiDiGraph
from numpy import array_split
from pandas import DataFrame, concat
from pydantic import BaseModel, InstanceOf, field_validator
from .utils import graphs, matrix_utils


class BuildsGrapher(BaseModel):
    city_osm_id: int
    city_crs: int
    public_transport_speeds: dict = None
    walk_speed: dict = None
    drive_speed: dict = None
    gdf_files: dict = None

    def get_intermodal_graph(self) -> MultiDiGraph:
        G_public_transport: MultiDiGraph = graphs.get_public_trasport_graph(
            self.city_osm_id, self.city_crs, self.gdf_files, self.public_transport_speeds
        )
        G_walk: MultiDiGraph = graphs.get_osmnx_graph(
            self.city_osm_id, self.city_crs, "walk", speed=self.walk_speed
        )

        G_drive: MultiDiGraph = graphs.get_osmnx_graph(
            self.city_osm_id, self.city_crs, "drive", speed=self.drive_speed
        )
        print("Union of city_graphs...")
        nx_intermodal_graph = graphs.graphs_spatial_union(G_walk, G_drive)
        if G_public_transport.number_of_edges() > 0:
            nx_intermodal_graph = graphs.graphs_spatial_union(
                nx_intermodal_graph, G_public_transport
            )

        for u, v, d in nx_intermodal_graph.edges(data=True):
            if "time_min" not in d:
                d["time_min"] = round(d["length_meter"] / G_walk.graph["walk speed"], 2)
            if "desc" not in d:
                d["desc"] = ""

        for u, d in nx_intermodal_graph.nodes(data=True):
            if "stop" not in d:
                d["stop"] = "False"
            if "desc" not in d:
                d["desc"] = ""

        nx_intermodal_graph.graph["graph_type"] = "intermodal graph"
        nx_intermodal_graph.graph["car speed"] = G_drive.graph["car speed"]
        nx_intermodal_graph.graph.update(
            {k: v for k, v in G_public_transport.graph.items() if "speed" in k}
        )
        print("Intermodal graph done!")
        return nx_intermodal_graph


class BuildsMatrixer(BaseModel):
    buildings_from: InstanceOf[GeoDataFrame]
    services_to: InstanceOf[GeoDataFrame]
    weight: str
    city_crs: int
    nx_intermodal_graph: InstanceOf[nx.MultiDiGraph]

    @field_validator('weight')
    @classmethod
    def ensure_weight(cls, v:Any):
        print(v)
        if v != "time_min" and v != "length_meter":
            raise ValueError("weight can only be 'time_min' or 'length_meter'")
        return v

    def get_adjacency_matrix(self) -> DataFrame:
        buildings_from = self.buildings_from.to_crs(self.city_crs)
        services_to = self.services_to.to_crs(self.city_crs)
        mobility_sub_graph = matrix_utils.get_subgraph(
            self.nx_intermodal_graph.copy(),
            "type",
            ["subway", "bus", "tram", "trolleybus", "walk"],
        )

        nk_graph = matrix_utils.convert_nx2nk(
            mobility_sub_graph,
            idmap=matrix_utils.get_nx2nk_idmap(mobility_sub_graph),
            weight=self.weight,
        )

        graph_with_geom = matrix_utils.load_graph_geometry(self.nx_intermodal_graph)
        df = DataFrame.from_dict(
            dict(graph_with_geom.nodes(data=True)), orient="index"
        )
        graph_gdf = GeoDataFrame(df, geometry=df["geometry"], crs=self.city_crs)
        from_houses = graph_gdf["geometry"].sindex.nearest(
            buildings_from["geometry"], return_distance=True, return_all=False
        )
        to_services = graph_gdf["geometry"].sindex.nearest(
            services_to["geometry"], return_distance=True, return_all=False
        )
        distance_matrix = DataFrame(
            0, index=to_services[0][1], columns=from_houses[0][1]
        )

        splited_matrix = array_split(
            distance_matrix.copy(deep=True),
            int(len(distance_matrix) / 1000) + 1,
        )

        # TODO use a* instead dijkstra
        for i in range(len(splited_matrix)):
            r = distance.SPSP(
                G=nk_graph, sources=splited_matrix[i].index.values
            ).run()
            splited_matrix[i] = splited_matrix[i].apply(
                lambda x: matrix_utils.get_nk_distances(r, x), axis=1
            )
            del r
        distance_matrix = concat(splited_matrix)
        del splited_matrix
        distance_matrix.index = services_to.index
        distance_matrix.columns = buildings_from.index
        return distance_matrix


class BuildsAvailabilitier(BaseModel):
    city_crs: int

    def get_accessibility_isochrone(self):
        crs = self.city_crs
        print("future func")
        return None
