"""Tests for Statcast zone assignment."""

from __future__ import annotations

import numpy as np

from src.statcast_zones import assign_statcast_zone, zone_rectangles


def test_inside_zone_center_is_five():
    zone = assign_statcast_zone(0.0, 2.5, 3.5, 1.5)
    assert zone == 5


def test_outside_quadrant_zones():
    assert assign_statcast_zone(-0.1, 4.0, 3.5, 1.5) == 11
    assert assign_statcast_zone(0.0, 4.0, 3.5, 1.5) == 12
    assert assign_statcast_zone(-0.1, 1.0, 3.5, 1.5) == 13
    assert assign_statcast_zone(0.0, 1.0, 3.5, 1.5) == 14
    assert assign_statcast_zone(-1.2, 3.0, 3.5, 1.5) == 11
    assert assign_statcast_zone(-1.2, 2.5, 3.5, 1.5) == 13
    assert assign_statcast_zone(1.2, 3.0, 3.5, 1.5) == 12
    assert assign_statcast_zone(1.2, 2.5, 3.5, 1.5) == 14
    assert assign_statcast_zone(-1.0, 1.25, 3.5, 1.5) == 13


def _rect_covering(rects, x: float, z: float) -> int | None:
    matches = [
        rect.zone
        for rect in rects
        if rect.x0 <= x <= rect.x1 and rect.z0 <= z <= rect.z1
    ]
    if not matches:
        return None
    for zone in matches:
        if 1 <= zone <= 9:
            return zone
    return matches[0]


def test_assignment_matches_zone_rectangles():
    rects = zone_rectangles(x_min=-1.5, x_max=1.5, z_min=0.5, z_max=4.5)
    samples = [
        (0.0, 2.5),
        (-0.2, 3.0),
        (0.2, 1.8),
        (-1.0, 1.25),
        (1.2, 3.8),
        (-1.2, 0.8),
        (0.01, 4.0),
    ]
    for x, z in samples:
        assert assign_statcast_zone(x, z, 3.5, 1.5) == _rect_covering(rects, x, z)


def test_zone_rectangles_cover_below_plate_corners():
    rects = zone_rectangles(x_min=-1.5, x_max=1.5, z_min=0.5, z_max=4.5)
    assert _rect_covering(rects, -1.0, 1.25) == 13
    assert assign_statcast_zone(-1.0, 1.25, 3.5, 1.5) == 13


def test_zone_rectangles_outer_quadrants():
    rects = zone_rectangles(x_min=-1.5, x_max=1.5, z_min=0.5, z_max=4.5)
    assert _rect_covering(rects, -1.2, 3.8) == 11
    assert _rect_covering(rects, 1.2, 3.8) == 12
    assert _rect_covering(rects, -1.2, 0.8) == 13
    assert _rect_covering(rects, 1.2, 0.8) == 14


def test_vectorized_assignment():
    zones = assign_statcast_zone(
        np.array([0.0, 0.0]),
        np.array([4.0, 2.5]),
        np.array([3.5, 3.5]),
        np.array([1.5, 1.5]),
    )
    assert zones.tolist() == [12, 5]


def test_zone_rectangles_cover_statcast_zones():
    rects = zone_rectangles()
    assert len(rects) == 17
    assert {rect.zone for rect in rects} == set(range(1, 10)) | {11, 12, 13, 14}
