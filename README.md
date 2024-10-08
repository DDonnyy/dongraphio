# Outdated. 
## It will no longer be supported. The latest version of the method has been migrated to [IduEdu](https://github.com/DDonnyy/IduEdu) . 

# DonGraphio

Small utility library containing graph algorighms used in other projects.

## Base usage example
```pip install dongraphio```
```python
import geopandas as gpd
from dongraphio import DonGraphio
from dongraphio import GraphType
import networkx as nx
from shapely import Point

dongrph = DonGraphio(city_crs=32638)
    
intermodal_graph = dongrph.get_intermodal_graph_from_osm(city_osm_id=3955288)
nx.write_graphml(intermodal_graph,"city_intermodal.graphml")

builds_from = gpd.read_file("test_data/buildings.geojson")
services_to = gpd.read_file("test_data/services.geojson")
adjacency_matrix = dongrph.get_adjacency_matrix(gdf_from=builds_from, gdf_to=services_to, weight="time_min",graph_type=[GraphType.PUBLIC_TRANSPORT, GraphType.WALK])
adjacency_matrix.to_csv("city_adjacency_matrix.csv")

accessibility_isochrones, public_transport_routes, public_transport_stops = dongrph.get_accessibility_isochrones(
    graph_type=[GraphType.PUBLIC_TRANSPORT, GraphType.WALK],
    points=Point(571747,5709639),
    weight_value=15,
    weight_type="time_min",
)
accessibility_isochrones.to_file("city_accessibility_isochrones.geojson")
public_transport_routes.to_file("city_public_transport.geojson")
public_transport_stops.to_file("city_public_stops.geojson")
```

To get rid of GeoPandas warning message about Shapely one can use following construction in their code:
```python
import os
os.environ["USE_PYGEOS"] = os.environ.get("USE_PYGEOS", "0")
```
