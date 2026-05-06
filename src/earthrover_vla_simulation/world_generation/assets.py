# assets.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RoomConfig:
    name: str
    template_world: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    wall_margin: float = 0.0
    spawn_clearance: float = 0.5


@dataclass(frozen=True)
class ObjectConfig:
    name: str
    uri: str
    size_x: float
    size_y: float
    min_spacing: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class WorldConfig:
    room: RoomConfig
    objects: list[ObjectConfig]


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


# loads config into World object
def load_world_config(config_path: str | Path) -> WorldConfig:
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # check correct yaml formatting
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping.")

    room_raw = _require(data, "room")
    objects_raw = _require(data, "objects")

    # check correct room and object formatting
    if not isinstance(room_raw, dict):
        raise ValueError("'room' must be a mapping.")
    if not isinstance(objects_raw, list):
        raise ValueError("'objects' must be a list.")

    # check correct region formatting
    regions = _require(room_raw, "regions")
    if not isinstance(regions, dict):
        raise ValueError("'room.regions' must be a mapping.")

    # assign the room
    room = RoomConfig(
        name=str(_require(room_raw, "name")),
        template_world=str(_require(room_raw, "template_world")),
        xmin=float(_require(regions, "xmin")),
        xmax=float(_require(regions, "xmax")),
        ymin=float(_require(regions, "ymin")),
        ymax=float(_require(regions, "ymax")),
        wall_margin=float(room_raw.get("wall_margin", 0.0)),
        spawn_clearance=float(room_raw.get("spawn_clearance",1.0))
        
    )

    # validate room min and maxes
    if room.xmin >= room.xmax or room.ymin >= room.ymax:
        raise ValueError("Room regions are invalid: min must be less than max.")


    objects: list[ObjectConfig] = []
    # validate and append objects to the objects list
    for i, obj_raw in enumerate(objects_raw):
        # raw object type check
        if not isinstance(obj_raw, dict):
            raise ValueError(f"objects[{i}] must be a mapping.")

        # validate object size
        size = _require(obj_raw, "size")
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError(f"objects[{i}].size must be a 2-element list.")

        objects.append(
            ObjectConfig(
                name=str(_require(obj_raw, "name")),
                uri=str(_require(obj_raw, "uri")),
                size_x=float(size[0]),
                size_y=float(size[1]),
                min_spacing=float(obj_raw.get("min_spacing", 0.0)),
                z=float(obj_raw.get("z", 0.0)),
            )
        )

    # minimum object size
    if not objects:
        raise ValueError("At least one object must be defined.")

    return WorldConfig(room=room, objects=objects)