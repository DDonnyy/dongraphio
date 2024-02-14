import time
from json import JSONDecodeError
import pandas as pd
import requests


def get_boundary(osm_id):

    overpass_url = "http://lz4.overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
            (
              relation({osm_id});
            );
    out geom;
    """
    result = requests.get(overpass_url, params={"data": overpass_query})
    json_result = result.json()

    return json_result


def get_routes(osm_id, public_transport_type):

    overpass_url = "http://lz4.overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
            (
                relation({osm_id});
            );map_to_area;
            (
                relation(area)['route'='{public_transport_type}'];
            );
    out geom;
    """
    result = requests.get(overpass_url, params={"data": overpass_query})
    json_result = result.json()["elements"]

    return pd.DataFrame(json_result)


def overpass_query(func, *args, attempts=5):

    for i in range(attempts):
        try:
            return func(*args)
        except JSONDecodeError:
            print("Another attempt to get response from Overpass API...")
            time.sleep(20)
            continue

    raise SystemError(
        """Something went wrong with Overpass API when JSON was parsed. Check the query and to send it later."""
    )
