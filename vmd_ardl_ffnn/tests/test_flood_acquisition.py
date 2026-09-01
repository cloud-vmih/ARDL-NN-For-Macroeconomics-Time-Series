from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("requests")

from vmd_ardl_ffnn.flood_acquisition import (
    _era5_objects,
    _interval_runoff_metres_to_mm,
)


def test_ncar_sro_interval_amounts_are_not_differenced() -> None:
    source = np.array([0.001, 0.003, 0.001, -1e-9])

    converted = _interval_runoff_metres_to_mm(source)

    np.testing.assert_allclose(converted, [1.0, 3.0, 1.0, 0.0])


def test_era5_year_object_names_cover_month_boundaries() -> None:
    objects = _era5_objects(2024)

    assert len(objects) == 24
    assert objects[0][1].parts[0] == "2024"
    assert objects[0][1].name.endswith("2024010106_2024011606.nc")
    assert objects[-1][1].name.endswith("2024121606_2025010106.nc")
