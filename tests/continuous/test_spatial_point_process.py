"""Test SpatialPointProcess."""
# flake8: noqa
import pytest
import numpy as np
from stochastic.continuous import SpatialPointProcess


def test_poisson_process_str_repr(density_1D, density_2D, density_3D, density_kwargs):
    for density in (density_1D, density_2D, density_3D):
        if (not callable(density)) or
           (not isinstance(density, (list, tuple, np.ndarray))) or
           (not isinstance(density_kwargs, dict)):
                with pytest.raises(ValueError):
                    instance = SpatialPointProcess(density, density_kwargs)
        else: 
            instance = SpatialPointProcess(density, density_kwargs)
            assert isinstance(repr(instance), str)
            assert isinstance(str(instance), str)

def test_poisson_process_sample(density_1D, density_2D, density_3D, n_fixture,
    bounds_1D, bounds_2D, bounds_3D, density_kwargs):
    instance = SpatialPointProcess(rate)
    if n_fixture is None and length is None:
        with pytest.raises(ValueError):
            s = instance.sample(n_fixture, length, zero)
    elif length is not None and n_fixture is None:
        s = instance.sample(n_fixture, length, zero)
        assert s[-1] >= length
    else:  # n_fixture is not None:
        s = instance.sample(n_fixture, length, zero)
        assert len(s) == n_fixture + int(zero)

def test_poisson_process_times(rate, n):
    instance = SpatialPointProcess(rate)
    with pytest.raises(AttributeError):
        times = instance.times(n)
