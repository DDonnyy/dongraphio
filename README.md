# DonGraphio

Small utility library containing graph algorighms used in other projects.

## Base usage example
```pip install dongraphio```
```python
import geopandas as gpd
import networkx as nx
from dongraphio import DonGraphio

builds = gpd.read_file("test_data/buildings.geojson")
services = gpd.read_file("test_data/services.geojson")

dongrph = DonGraphio(city_osm_id=7226665, city_crs=32643)

graph = dongrph.get_intermodal_graph()
nx.write_graphml(graph,"test_data/test.graph")

matrix = dongrph.get_adjacency_matrix(buildings_from=builds, services_to=services, weight="time_min")
print(matrix)

accessibility_isochrone = dongrph.get_accessibility_isochrone() # COMING SOON!
```

To get rid of GeoPandas warning message about Shapely one can use following construction in their code:
```python
import os
os.environ["USE_PYGEOS"] = os.environ.get("USE_PYGEOS", "0")
```
