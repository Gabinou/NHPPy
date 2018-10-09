"""Test SpatialPointProcess."""
# flake8: noqa
import pytest
import numpy as np
from stochastic.continuous import SpatialPointProcess


def test_poisson_process_str_repr(density_1D, density_2D, density_3D,
                                  density_kwargs):
    for density in (density_1D, density_2D, density_3D):
        if ((not callable(density)) or
           (not isinstance(density, (list, tuple, np.ndarray))) or
           (not isinstance(density_kwargs, dict))):
            with pytest.raises(ValueError):
                instance = SpatialPointProcess(density, density_kwargs)
        else:
            instance = SpatialPointProcess(density, density_kwargs)
            assert isinstance(repr(instance), str)
            assert isinstance(str(instance), str)


def test_poisson_process_sample(density_1D, density_2D, density_3D,
                                n_fixture, bounds_1D, bounds_2D, bounds_3D,
                                density_kwargs):
    bounds = (bounds_1D, bounds_2D, bounds_3D,)
    density = (density_1D, density_2D, density_3D,)
    for bound_index in range(3):
        for density_index in range(3):
            if ((isinstance(density[density_index], (list, tuple, np.ndarray)))
               and (bound_index != density_index)):
                with pytest.raises(ValueError):
                    instance = SpatialPointProcess(density[density_index],
                                                   density_kwargs)
            elif (not isinstance(density_kwargs, dict) or
                  not isinstance(density, (list, tuple, np.ndarray))):
                with pytest.raises(ValueError):
                    instance = SpatialPointProcess(density[density_index],
                                                   density_kwargs)
            else:
                instance = SpatialPointProcess(density[density_index],
                                               density_kwargs)
            if ((not isinstance(n_fixture, int)) or
                (not isinstance(bounds, (list, tuple, np.ndarray))) or
                (callable(density[density_index]) and
                    (density_index != bound_index))):
                with pytest.raises(ValueError):
                    s = instance.sample(n=n_fixture,
                                        bounds=bounds[bound_index])
            else:
                s = instance.sample(n=n_fixture, bounds=bounds[bound_index])
                assert len(s) == n_fixture + int(zero)


def test_poisson_process_times(density_1D, density_2D, density_3D,
                                n_fixture, bounds_1D, bounds_2D, bounds_3D,
                                density_kwargs):
    bounds = (bounds_1D, bounds_2D, bounds_3D,)
    density = (density_1D, density_2D, density_3D,)
    for bound_index in range(3):
        for density_index in range(3):
            if ((isinstance(density[density_index], (list, tuple, np.ndarray)))
               and (bound_index != density_index)):
                with pytest.raises(ValueError):
                    instance = SpatialPointProcess(density[density_index],
                                                   density_kwargs)
            elif (not isinstance(density_kwargs, dict) or
                  not isinstance(density, (list, tuple, np.ndarray))):
                with pytest.raises(ValueError):
                    instance = SpatialPointProcess(density[density_index],
                                                   density_kwargs)
            else:
                instance = SpatialPointProcess(density[density_index],
                                               density_kwargs)
                with pytest.raises(AttributeError):
                    times = instance.times(n)
