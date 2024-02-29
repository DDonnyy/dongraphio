# DonGraphio

Small utility library containing graph algorighms used in other projects.

## Base usage example
```pip install dongraphio```
```python
import geopandas as gpd
from dongraphio import DonGraphio
from dongraphio import GraphType

builds = gpd.read_file("test_data/buildings.geojson")
services = gpd.read_file("test_data/services.geojson")

dongrph = DonGraphio(city_crs=32636)

intermodal_graph = dongrph.get_intermodal_graph_from_osm(city_osm_id=421007)

adjacency_matrix = dongrph.get_adjacency_matrix(buildings_from=builds, services_to=services, weight="time_min")

accessibility_isochrones, public_transport_routes, public_transport_stops = dongrph.get_accessibility_isochrones(
    graph_type=[GraphType.PUBLIC_TRANSPORT, GraphType.WALK],
    x_from=348925,
    y_from=6648260,
    weight_value=20,
    weight_type="time_min",
)
```

To get rid of GeoPandas warning message about Shapely one can use following construction in their code:
```python
import os
os.environ["USE_PYGEOS"] = os.environ.get("USE_PYGEOS", "0")
```
