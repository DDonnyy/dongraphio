from geopandas import GeoDataFrame
from networkit import distance
from networkx import MultiDiGraph
from numpy import array_split
from pandas import DataFrame, concat
from pydantic import BaseModel, InstanceOf
from src.city_grapher.utils import graphs, matrix_utils


class CityGrapher(BaseModel):
    city_osm_id: int
    city_crs: int
    _nx_intermodal_graph: InstanceOf[MultiDiGraph]

    def get_intermodal_graph(
        self,
        public_transport_speeds=None,
        walk_speed=None,
        drive_speed=None,
        gdf_files=None,
    ) -> MultiDiGraph:
        G_public_transport: MultiDiGraph = graphs.get_public_trasport_graph(
            self.city_osm_id, self.city_crs, gdf_files, public_transport_speeds
        )
        G_walk: MultiDiGraph = graphs.get_osmnx_graph(
            self.city_osm_id, self.city_crs, "walk", speed=walk_speed
        )

        G_drive: MultiDiGraph = graphs.get_osmnx_graph(
            self.city_osm_id, self.city_crs, "drive", speed=drive_speed
        )
        print("Union of city_graphs...")
        self._nx_intermodal_graph = graphs.graphs_spatial_union(G_walk, G_drive)
        if G_public_transport.number_of_edges() > 0:
            self._nx_intermodal_graph = graphs.graphs_spatial_union(
                self._nx_intermodal_graph, G_public_transport
            )

        for u, v, d in self._nx_intermodal_graph.edges(data=True):
            if "time_min" not in d:
                d["time_min"] = round(d["length_meter"] / G_walk.graph["walk speed"], 2)
            if "desc" not in d:
                d["desc"] = ""

        for u, d in self._nx_intermodal_graph.nodes(data=True):
            if "stop" not in d:
                d["stop"] = "False"
            if "desc" not in d:
                d["desc"] = ""

        self._nx_intermodal_graph.graph["graph_type"] = "intermodal graph"
        self._nx_intermodal_graph.graph["car speed"] = G_drive.graph["car speed"]
        self._nx_intermodal_graph.graph.update(
            {k: v for k, v in G_public_transport.graph.items() if "speed" in k}
        )
        print("Intermodal graph done!")
        return self._nx_intermodal_graph

    def get_adjacency_matrix(
        self,
        buildings_from: GeoDataFrame,
        services_to: GeoDataFrame,
        weight="length_meter",
    ) -> DataFrame:
        buildings_from = buildings_from.to_crs(self.city_crs)
        services_to = services_to.to_crs(self.city_crs)
        mobility_sub_graph = matrix_utils.get_subgraph(
            self._nx_intermodal_graph.copy(),
            "type",
            ["subway", "bus", "tram", "trolleybus", "walk"],
        )

        nk_graph = matrix_utils.convert_nx2nk(
            mobility_sub_graph,
            idmap=matrix_utils.get_nx2nk_idmap(mobility_sub_graph),
            weight=weight,
        )

        graph_with_geom = matrix_utils.load_graph_geometry(self._nx_intermodal_graph)
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
