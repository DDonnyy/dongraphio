from typing import Optional

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import osm2geojson
import osmnx.graph as ox_graph
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from shapely import LineString, Point, geometry
from tqdm.auto import tqdm

from ..utils import (
    get_boundary,
    get_nearest_edge_geometry,
    get_routes,
    overpass_request,
    parse_overpass_route_response,
    project_platforms,
    project_point_on_edge,
    update_edges,
)

pd.options.mode.chained_assignment = None
tqdm.pandas()


class BuildsGrapher(BaseModel):
    city_osm_id: int
    city_crs: int
    keep_city_boundary: bool = True
    public_transport_speeds: Optional[dict] = None
    walk_speed: Optional[dict] = None
    drive_speed: Optional[dict] = None
    gdf_files: Optional[dict] = None
    truncate_by_edge: Optional[bool] = False

    def get_intermodal_graph(self) -> nx.MultiDiGraph:
        G_public_transport = self.get_public_trasport_graph()

        G_walk: nx.MultiDiGraph = self.get_osmnx_graph("walk", self.walk_speed)

        G_drive: nx.MultiDiGraph = self.get_osmnx_graph("drive", self.drive_speed)
        logger.debug("Preparing union of city_graphs...")
        nx_intermodal_graph: nx.MultiDiGraph = self.graphs_spatial_union(G_walk, G_drive)
        if G_public_transport.number_of_edges() > 0:
            nx_intermodal_graph: nx.MultiDiGraph = self.graphs_spatial_union(nx_intermodal_graph, G_public_transport)

        for _u, _v, d in nx_intermodal_graph.edges(data=True):
            if "time_min" not in d:
                d["time_min"] = round(d["length_meter"] / G_walk.graph["walk speed"], 2)
            if "desc" not in d:
                d["desc"] = ""

        for _u, d in nx_intermodal_graph.nodes(data=True):
            if "stop" not in d:
                d["stop"] = "False"
            if "desc" not in d:
                d["desc"] = ""

        nx_intermodal_graph.graph["graph_type"] = "intermodal graph"
        nx_intermodal_graph.graph["car speed"] = G_drive.graph["car speed"]
        nx_intermodal_graph.graph.update({k: v for k, v in G_public_transport.graph.items() if "speed" in k})
        logger.info("Intermodal graph done!\n")
        return nx_intermodal_graph

    def join_graph(
        self, G_base: nx.MultiDiGraph, G_to_project: nx.MultiDiGraph, points_df: pd.DataFrame
    ) -> nx.MultiDiGraph:

        new_nodes = points_df.set_index("node_id_to_project")["connecting_node_id"]
        for n1, n2, d in tqdm(
            G_to_project.edges(data=True),
            desc=f"Joining {G_base.graph.get('graph_type')} and {G_to_project.graph.get('graph_type')}",
            leave=False,
        ):
            G_base.add_edge(int(new_nodes[n1]), int(new_nodes[n2]), **d)
            nx.set_node_attributes(
                G_base,
                {
                    int(new_nodes[n1]): G_to_project.nodes[n1],
                    int(new_nodes[n2]): G_to_project.nodes[n2],
                },
            )

        return G_base

    def get_osmnx_graph(self, graph_type: str, speed: Optional[dict]) -> nx.MultiDiGraph:
        boundary = overpass_request(get_boundary, self.city_osm_id)
        boundary = osm2geojson.json2geojson(boundary)
        boundary = gpd.GeoDataFrame.from_features(boundary["features"]).set_crs(4326)

        logger.debug("Extracting and preparing {} graph from OSM ...", graph_type)
        G_ox = ox_graph.graph_from_polygon(
            polygon=boundary["geometry"][0], network_type=graph_type, truncate_by_edge=self.truncate_by_edge
        )
        G_ox.graph["approach"] = "primal"

        nodes: gpd.GeoDataFrame
        edges: gpd.GeoDataFrame
        nodes, edges = momepy.nx_to_gdf(G_ox, points=True, lines=True, spatial_weights=False)
        nodes = nodes.to_crs(self.city_crs).set_index("nodeID")
        nodes_coord = nodes.geometry.apply(
            lambda p: {"x": round(p.coords[0][0], 2), "y": round(p.coords[0][1], 2)}
        ).to_dict()

        edges = edges[["length", "node_start", "node_end", "geometry"]].to_crs(self.city_crs)
        edges["type"] = graph_type
        edges["geometry"] = edges["geometry"].apply(
            lambda x: LineString([tuple(round(c, 2) for c in n) for n in x.coords] if x else None)
        )

        travel_type = "walk" if graph_type == "walk" else "car"
        if not speed:
            speed = 4 * 1000 / 60 if graph_type == "walk" else 17 * 1000 / 60
        G = nx.MultiDiGraph()
        for _, edge in tqdm(edges.iterrows(), total=len(edges), desc=f"Collecting {graph_type} graph", leave=False):
            p1 = int(edge.node_start)
            p2 = int(edge.node_end)
            geom = (
                LineString(
                    (
                        [
                            (nodes_coord[p1]["x"], nodes_coord[p1]["y"]),
                            (nodes_coord[p2]["x"], nodes_coord[p2]["y"]),
                        ]
                    )
                )
                if not edge.geometry
                else edge.geometry
            )
            G.add_edge(
                p1,
                p2,
                length_meter=edge.length,
                geometry=str(geom),
                type=travel_type,
                time_min=round(edge.length / speed, 2),
            )

        nx.set_node_attributes(G, nodes_coord)
        G.graph["crs"] = "epsg:" + str(self.city_crs)
        G.graph["graph_type"] = travel_type + " graph"
        G.graph[travel_type + " speed"] = round(speed, 2)

        return G

    def public_routes_to_edges(self, transport_type, speed, boundary, use_boundary: bool):
        routes = overpass_request(get_routes, self.city_osm_id, transport_type)
        try:
            tqdm.pandas(desc=f"Extracting and preparing {transport_type} routes from OSM", postfix=None, leave=False)
            df_routes: gpd.GeoDataFrame = routes.progress_apply(
                lambda x: parse_overpass_route_response(x, self.city_crs, boundary, use_boundary),
                axis=1,
                result_type="expand",
            )
            df_routes = gpd.GeoDataFrame(df_routes).dropna(subset=["way"]).set_geometry("way")

        except (KeyError, ZeroDivisionError):
            logger.warning(
                "It seems there are no {} routes in the city. This transport type will be skipped.", transport_type
            )
            return []

        # some stops don't lie on lines, therefore it's needed to project them
        stop_points = df_routes.apply(lambda x: project_platforms(x, self.city_crs), axis=1)

        edges = []
        time_on_stop = 1
        for i, route in stop_points.iterrows():
            length = np.diff(list(route["distance"]))
            for j in range(len(route["pathes"])):
                edge_length = float(length[j])
                p1 = route["pathes"][j][0]
                p2 = route["pathes"][j][1]
                d = {
                    "time_min": round(edge_length / speed + time_on_stop, 2),
                    "length_meter": round(edge_length, 2),
                    "type": transport_type,
                    "desc": f"route {i}",
                    "geometry": str(LineString([p1, p2])),
                }
                edges.append((p1, p2, d))

        return edges

    def graphs_spatial_union(
        self, G_base: nx.MultiDiGraph, G_to_project: nx.MultiDiGraph
    ) -> nx.MultiDiGraph:  # pylint: disable=invalid-name
        points = gpd.GeoDataFrame(
            [[n, Point((d["x"], d["y"]))] for n, d in G_to_project.nodes(data=True)],
            columns=["node_id_to_project", "geometry"],
        )
        edges_geom = get_nearest_edge_geometry(points, G_base)
        projected_point_info = project_point_on_edge(edges_geom)
        check_point_on_line = projected_point_info.apply(
            lambda x: x.edge_geometry.buffer(1).contains(x.nearest_point_geometry), axis=1
        ).all()
        if not check_point_on_line:
            raise ValueError("Some projected points don't lie on edges")
        points_on_lines = projected_point_info[
            (projected_point_info["len_from_start"] != 0) & (projected_point_info["len_to_end"] != 0)
        ]

        points_on_points = projected_point_info[~projected_point_info.index.isin(points_on_lines.index)]
        try:
            points_on_points["connecting_node_id"] = points_on_points.apply(
                lambda x: x.edge_id[0] if x.len_from_start == 0 else x.edge_id[1], axis=1
            )
        except ValueError:
            logger.warning("No matching nodes were detected, seems like your data is not the same as in OSM.")

        updated_G_base, points_on_lines = update_edges(points_on_lines, G_base)  # pylint: disable=invalid-name
        points_df = pd.concat([points_on_lines, points_on_points])
        united_graph = self.join_graph(updated_G_base, G_to_project, points_df)
        return united_graph

    def get_public_trasport_graph(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        edegs_different_types = []
        if not self.public_transport_speeds:
            self.public_transport_speeds = {  # TODO add enum in future version
                "subway": 12 * 1000 / 60,
                "tram": 15 * 1000 / 60,
                "trolleybus": 12 * 1000 / 60,
                "bus": 17 * 1000 / 60,
            }
        from_file = False  # TODO REMAKE FILE MANAGEMENT SYSTEM

        # for transport in gdf_files.values():
        #     if transport.get("stops") or transport.get("routes"):
        #         from_file = True

        if not from_file:
            logger.warning(
                "Files with routes or with stops was not found. The graph will be built based on data from OSM"
            )
            boundary = overpass_request(get_boundary, self.city_osm_id)
            boundary = osm2geojson.json2geojson(boundary)
            boundary = geometry.shape(boundary["features"][0]["geometry"])

            for transport_type, speed in self.public_transport_speeds.items():
                edges = self.public_routes_to_edges(transport_type, speed, boundary, not self.truncate_by_edge)
                edegs_different_types.extend(edges)
        # else:
        #     logger.info("Getting public routes data from files %s", ", ".join(gdf_files.values()))
        #     for transport_type, speed in self.public_transport_speeds.items():
        #         files = gdf_files.get(transport_type)
        #         if not files.get("routes") or not files.get("stops"):
        #             logger.warning('No data provided for "{transport_type}", skipping this transport type')
        #             continue
        #         edges = self.public_routes_to_edges_from_file(city_crs, transport_type, speed, files)
        #         edegs_different_types.extend(edges)

        graph.add_edges_from(edegs_different_types)
        if len(edegs_different_types) == 0:
            logger.warning("No data found for public transport, this graph will be empty.")
            return graph

        node_attributes = {
            node: {
                "x": round(node[0], 2),
                "y": round(node[1], 2),
                "stop": "True",
                "desc": [],
            }
            for node in list(graph.nodes)
        }

        for p1, p2, data in list(graph.edges(data=True)):
            transport_type = data["type"]
            node_attributes[p1]["desc"].append(transport_type)
            node_attributes[p2]["desc"].append(transport_type)

        for data in node_attributes.values():
            data["desc"] = ", ".join(set(data["desc"]))
        nx.set_node_attributes(graph, node_attributes)
        graph = nx.convert_node_labels_to_integers(graph)
        graph.graph["crs"] = "epsg:" + str(self.city_crs)
        graph.graph["graph_type"] = "public transport graph"
        graph.graph.update({k + " speed": round(v, 2) for k, v in self.public_transport_speeds.items()})

        logger.info("Public transport graph done!")
        return graph

    def public_routes_to_edges_from_file(
        self, city_crs: int, transport_type: str, speed: int | float, gdf_files: dict[str, str]
    ) -> list[tuple[int, int, dict[str, int | str]]]:
        edges = []
        try:
            gdf_stops = gpd.read_file(gdf_files.get("stops"))

            gdf_routes = gpd.read_file(gdf_files.get("routes"))
            ways: gpd.GeoDataFrame = gdf_routes[["route", "geometry"]].copy()
            ways = ways.explode(index_parts=False).to_crs(city_crs)
            ways["route"] = ways["route"].apply(lambda x: str(x).strip())
            ways.set_index("route", inplace=True)

            platforms = pd.DataFrame(columns=["route", "geometry"])

            for index, row in gdf_stops.iterrows():
                for i in str(row["route"]).split(","):
                    df = pd.DataFrame(({i: row["geometry"]}).items(), columns=["route", "geometry"])
                    platforms = pd.concat([platforms, df])
            platforms.reset_index(inplace=True, drop=True)
            platforms["route"] = platforms["route"].apply(lambda x: str(x).strip())
            df_routes = pd.DataFrame()

            for index, row in ways.iterrows():
                platforms_: pd.Series = (
                    platforms[platforms["route"] == str(index)].drop("route", axis=1).reset_index(drop=True).squeeze()
                )
                series = pd.Series({"way": row["geometry"], "platforms": platforms_})
                df_routes = pd.concat([df_routes, series], axis=1, ignore_index=True)
            df_routes = df_routes.transpose()
            df_routes = gpd.GeoDataFrame(df_routes).set_geometry("way")

            stop_points = df_routes.apply(lambda x: project_platforms(x, city_crs), axis=1)

            time_on_stop = 1
            for i, route in stop_points.iterrows():
                length = np.diff(list(route["distance"]))
                for j in range(len(route["pathes"])):
                    edge_length = float(length[j])
                    p1 = route["pathes"][j][0]
                    p2 = route["pathes"][j][1]
                    d = {
                        "time_min": round(edge_length / speed + time_on_stop, 2),
                        "length_meter": round(edge_length, 2),
                        "type": transport_type,
                        "desc": f"route {i}",
                        "geometry": str(LineString([p1, p2])),
                    }
                    edges.append((p1, p2, d))
        except KeyError:
            logger.error(
                "The 'route' column was not found in one of the files for '%s' . Please check their contents.",
                transport_type,
            )
        except Exception as err:  # pylint: disable=broad-except
            logger.error('File with routes or with stops was not found for "%s", error: %s', transport_type, repr(err))
        return edges
