# Hello and very well!!

### Base usage
```py
import geopandas as gpd
import networkx as nx
from citygrapher import Invoker

builds = gpd.read_file("test_data/buildings.geojson")
services = gpd.read_file("test_data/tara_kinder.geojson")

invoker = Invoker(city_osm_id=7226665, city_crs=32643)

graph = invoker.get_intermodal_graph()
nx.write_graphml(graph,"test_data/test.graph")

matrix = invoker.get_adjacency_matrix(buildings_from=builds,services_to=services,weight="time_min")
print(matrix)

accessibility_isochrone = invoker.get_accessibility_isochrone() # COMING SOON!

```