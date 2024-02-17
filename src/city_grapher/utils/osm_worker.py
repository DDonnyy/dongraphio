import logging
import os
import time
from json import JSONDecodeError

import geopandas as gpd
import pandas as pd
import requests

DEFAULT_OVERPASS_URL = os.environ.get("OVERPASS_URL", "http://lz4.overpass-api.de/api/interpreter")


def get_boundary(osm_id: int, overpass_url: str = DEFAULT_OVERPASS_URL, timeout: int = 300):

    overpass_query = f"""
    [out:json];
            (
                relation({osm_id});
            );
    out geom;
    """
    result = requests.get(overpass_url, params={"data": overpass_query}, timeout=timeout)
    json_result = result.json()

    return json_result


def get_routes(osm_id: int, public_transport_type: str, overpass_url: str = DEFAULT_OVERPASS_URL, timeout: int = 300):

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
    result = requests.get(overpass_url, params={"data": overpass_query}, timeout=timeout)
    json_result = result.json()["elements"]

    return pd.DataFrame(json_result)


def overpass_request(func, *args, attempts: int = 5, wait_time: int = 20) -> gpd.GeoDataFrame:

    for _ in range(attempts):
        try:
            return func(*args)
        except JSONDecodeError as exc:
            logging.debug("Got error on parsing Overpass response: '%s'", repr(exc))
            logging.info("Another attempt to get response from Overpass API in %d seconds...", wait_time)
            time.sleep(wait_time)

    raise SystemError(
        """Something went wrong with Overpass API when JSON was parsed. Check the query and to send it later."""
    )
