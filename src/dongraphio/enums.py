from enum import Enum


class EdgeTypes(Enum):
    SUBWAY = "subway"
    BUS = "bus"
    TRAM = "tram"
    TROLLEYBUS = "trolleybus"
    WALK = "walk"
    DRIVE = "car"


class GraphType(Enum):
    PUBLIC_TRANSPORT = {
        "name": "Общественный транспорт",
        "types": [EdgeTypes.SUBWAY, EdgeTypes.BUS, EdgeTypes.TRAM, EdgeTypes.TROLLEYBUS],
    }
    DRIVE = {"name": "Личный транспорт", "types": [EdgeTypes.DRIVE]}
    WALK = {"name": "Пешком", "types": [EdgeTypes.WALK]}
