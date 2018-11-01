"""Test SpatialPointProcess."""
# flake8: noqa
import pytest
import numpy as np
from stochastic.continuous import SpatialPointProcess

def test_spatial_point_process_str_repr_1D(density_1D, density_kwargs):
    if (((not callable(density_1D)) and
        (not isinstance(density_1D, (list, tuple, np.ndarray)))) or
        (not isinstance(density_kwargs, dict))):
        with pytest.raises(ValueError):
            instance = SpatialPointProcess(density_1D, density_kwargs)
    else:
        instance = SpatialPointProcess(density_1D, density_kwargs)
        assert isinstance(repr(instance), str)
        assert isinstance(str(instance), str)

def test_spatial_point_process_str_repr_2D(density_2D, density_kwargs):
    if (((not callable(density_2D)) and
        (not isinstance(density_2D, (list, tuple, np.ndarray)))) or
        (not isinstance(density_kwargs, dict))):
        with pytest.raises(ValueError):
            instance = SpatialPointProcess(density_2D, density_kwargs)
    else:
        instance = SpatialPointProcess(density_2D, density_kwargs)
        assert isinstance(repr(instance), str)
        assert isinstance(str(instance), str)

def test_spatial_point_process_str_repr_3D(density_3D, density_kwargs):
    if (((not callable(density_3D)) and
        (not isinstance(density_3D, (list, tuple, np.ndarray)))) or
        (not isinstance(density_kwargs, dict))):
        with pytest.raises(ValueError):
            instance = SpatialPointProcess(density_3D, density_kwargs)
    else:
        instance = SpatialPointProcess(density_3D, density_kwargs)
        assert isinstance(repr(instance), str)
        assert isinstance(str(instance), str)

def test_spatial_point_process_sample_1D(density_1D, n_fixture, bounds_1D, density_kwargs):
    if (((not callable(density_1D)) and
        (not isinstance(density_1D, (list, tuple, np.ndarray)))) or
        (not isinstance(density_kwargs, dict))):
        with pytest.raises(ValueError):
            instance = SpatialPointProcess(density_1D, density_kwargs)
    else:
        instance = SpatialPointProcess(density_1D, density_kwargs)
        if n_fixture is None or bounds_1D is None or bounds_1D is ():
            with pytest.raises(ValueError):
                s = instance.sample(n_fixture, bounds_1D)
        elif bounds_1D is not None and n_fixture is None:
            s = instance.sample(n_fixture, bounds_1D)
            assert s[-1] >= length
        else:
            s = instance.sample(n_fixture, bounds_1D)
            assert len(s) == n_fixture

def test_spatial_point_process_sample_2D(density_2D, n_fixture, bounds_2D, density_kwargs):
    if (((not callable(density_2D)) and
        (not isinstance(density_2D, (list, tuple, np.ndarray)))) or
        (not isinstance(density_kwargs, dict))):
        with pytest.raises(ValueError):
            instance = SpatialPointProcess(density_2D, density_kwargs)
    else:
        instance = SpatialPointProcess(density_2D, density_kwargs)
        if n_fixture is None or bounds_2D is None or bounds_2D is ():
            with pytest.raises(ValueError):
                s = instance.sample(n_fixture, bounds_2D)
        elif bounds_2D is not None and n_fixture is None:
            s = instance.sample(n_fixture, bounds_2D)
            assert s[-1] >= length
        else:
            s = instance.sample(n_fixture, bounds_2D)
            assert len(s) == n_fixture

def test_spatial_point_process_sample_3D(density_3D, n_fixture, bounds_3D, density_kwargs):
    if (((not callable(density_3D)) and
        (not isinstance(density_3D, (list, tuple, np.ndarray)))) or
        (not isinstance(density_kwargs, dict))):
        with pytest.raises(ValueError):
            instance = SpatialPointProcess(density_3D, density_kwargs)
    else:
        instance = SpatialPointProcess(density_3D, density_kwargs)
        if n_fixture is None or bounds_3D is None or bounds_3D is ():
            with pytest.raises(ValueError):
                s = instance.sample(n_fixture, bounds_3D)
        elif bounds_3D is not None and n_fixture is None:
            s = instance.sample(n_fixture, bounds_3D)
            assert s[-1] >= length
        else:
            s = instance.sample(n_fixture, bounds_3D)
            assert len(s) == n_fixture

def test_poisson_process_times(density_1D, density_kwargs, n):
    if (((not callable(density_1D)) and
        (not isinstance(density_1D, (list, tuple, np.ndarray)))) or
        (not isinstance(density_kwargs, dict))):
        with pytest.raises(ValueError):
            instance = SpatialPointProcess(density_1D, density_kwargs)
    else:
        instance = SpatialPointProcess(density_1D, density_kwargs)
        with pytest.raises(AttributeError):
            times = instance.times(n)

def test_poisson_process_times(density_2D, density_kwargs, n):
    if (((not callable(density_2D)) and
        (not isinstance(density_2D, (list, tuple, np.ndarray)))) or
        (not isinstance(density_kwargs, dict))):
        with pytest.raises(ValueError):
            instance = SpatialPointProcess(density_2D, density_kwargs)
    else:
        instance = SpatialPointProcess(density_2D, density_kwargs)
        with pytest.raises(AttributeError):
            times = instance.times(n)

def test_poisson_process_times(density_3D, density_kwargs, n):
    if (((not callable(density_3D)) and
        (not isinstance(density_3D, (list, tuple, np.ndarray)))) or
        (not isinstance(density_kwargs, dict))):
        with pytest.raises(ValueError):
            instance = SpatialPointProcess(density_3D, density_kwargs)
    else:
        instance = SpatialPointProcess(density_3D, density_kwargs)
        with pytest.raises(AttributeError):
            times = instance.times(n)