from enum import Enum


class EdgeTypes(Enum):
    """
    Enumeration class for edge types in graphs.
    """

    SUBWAY = "subway"
    BUS = "bus"
    TRAM = "tram"
    TROLLEYBUS = "trolleybus"
    WALK = "walk"
    DRIVE = "car"


class GraphType(Enum):
    """
    Enumeration class for possible graph types, containing Russian name and possible EdgeTypes(Enum).
    """

    PUBLIC_TRANSPORT = "public_transport"
    DRIVE = "drive"
    WALK = "walk"

    @property
    def russian_name(self) -> str:
        names = {
            GraphType.PUBLIC_TRANSPORT: "Общественный транспорт",
            GraphType.DRIVE: "Личный транспорт",
            GraphType.WALK: "Пешком",
        }
        return names[self]

    @property
    def edges(self):
        edges = {
            GraphType.PUBLIC_TRANSPORT: [EdgeTypes.SUBWAY, EdgeTypes.BUS, EdgeTypes.TRAM, EdgeTypes.TROLLEYBUS],
            GraphType.DRIVE: [EdgeTypes.DRIVE],
            GraphType.WALK: [EdgeTypes.WALK],
        }
        return edges[self]
