# layout.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

from assets import ObjectConfig, RoomConfig


@dataclass(frozen=True)
class Placement:
    object_name: str
    uri: str
    x: float
    y: float
    z: float
    yaw: float
    size_x: float
    size_y: float
    min_spacing: float


@dataclass(frozen=True)
class OrientedRect:
    cx: float
    cy: float
    hx: float
    hy: float
    yaw: float


def _rotate_point(px: float, py: float, yaw: float) -> tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * px - s * py, s * px + c * py)


def rect_corners(rect: OrientedRect) -> list[tuple[float, float]]:
    local = [
        (-rect.hx, -rect.hy),
        ( rect.hx, -rect.hy),
        ( rect.hx,  rect.hy),
        (-rect.hx,  rect.hy),
    ]
    corners: list[tuple[float, float]] = []
    for lx, ly in local:
        rx, ry = _rotate_point(lx, ly, rect.yaw)
        corners.append((rect.cx + rx, rect.cy + ry))
    return corners


def rect_axes(rect: OrientedRect) -> list[tuple[float, float]]:
    c = math.cos(rect.yaw)
    s = math.sin(rect.yaw)
    return [(c, s), (-s, c)]


def project_polygon(points: Iterable[tuple[float, float]], axis: tuple[float, float]) -> tuple[float, float]:
    ax, ay = axis
    vals = [px * ax + py * ay for px, py in points]
    return min(vals), max(vals)


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def rects_overlap(a: OrientedRect, b: OrientedRect) -> bool:
    a_pts = rect_corners(a)
    b_pts = rect_corners(b)

    for axis in rect_axes(a) + rect_axes(b):
        proj_a = project_polygon(a_pts, axis)
        proj_b = project_polygon(b_pts, axis)
        if not intervals_overlap(proj_a, proj_b):
            return False
    return True


def placement_to_rect(p: Placement) -> OrientedRect:
    inflate = p.min_spacing
    return OrientedRect(
        cx=p.x,
        cy=p.y,
        hx=(p.size_x / 2.0) + inflate,
        hy=(p.size_y / 2.0) + inflate,
        yaw=p.yaw,
    )


def fits_in_room(room: RoomConfig, placement: Placement) -> bool:
    rect = placement_to_rect(placement)
    for x, y in rect_corners(rect):
        if x < room.xmin + room.wall_margin:
            return False
        if x > room.xmax - room.wall_margin:
            return False
        if y < room.ymin + room.wall_margin:
            return False
        if y > room.ymax - room.wall_margin:
            return False
    return True


def collides_with_any(candidate: Placement, placed: list[Placement]) -> bool:
    cand_rect = placement_to_rect(candidate)
    for other in placed:
        other_rect = placement_to_rect(other)
        if rects_overlap(cand_rect, other_rect):
            return True
    return False


def sample_candidate(room: RoomConfig, obj: ObjectConfig, rng: random.Random) -> Placement:
    x = rng.uniform(room.xmin, room.xmax)
    y = rng.uniform(room.ymin, room.ymax)
    yaw = rng.uniform(-math.pi, math.pi)

    return Placement(
        object_name=obj.name,
        uri=obj.uri,
        x=x,
        y=y,
        z=obj.z,
        yaw=yaw,
        size_x=obj.size_x,
        size_y=obj.size_y,
        min_spacing=obj.min_spacing,
    )


def generate_layout(
    room: RoomConfig,
    objects: list[ObjectConfig],
    count: int,
    seed: int | None = None,
    max_attempts_per_object: int = 200,
) -> list[Placement]:
    rng = random.Random(seed)
    placed: list[Placement] = []

    for _ in range(count):
        success = False
        for _attempt in range(max_attempts_per_object):
            obj = rng.choice(objects)
            candidate = sample_candidate(room, obj, rng)

            if not fits_in_room(room, candidate):
                continue
            if collides_with_any(candidate, placed):
                continue

            placed.append(candidate)
            success = True
            break

        if not success:
            raise RuntimeError(
                "Failed to place all objects without overlap. "
                "Try reducing count, shrinking object sizes, or increasing room size."
            )

    return placed