import numpy as np

import utils
from base import Camera

from typing import Any, List, Tuple
from nptyping import NDArray


class Base(utils.Optimizable):

    def sample(self, uv: NDArray[(3, Any), float]) \
            -> Tuple[NDArray[(Any,), float], NDArray[(Any,), bool]]:
        '''Sample vignetting for a given pixel'''
        assert uv.shape[0] == 3, '`uv` must be homogeneous coordinates'
        assert np.all(uv[2, :] == 1), '`uv` must be pixel coordinates'
        pass


class Constant(Base):
    def __init__(self,
                 camera: Camera,
                 value: float) \
            -> None:
        self.camera = camera
        self.value = value

    def _get_params(self) -> List:
        return []

    def _set_params(self, a: List) -> None:
        pass

    def _get_lower_bound(self) -> NDArray:
        return np.empty(0)

    def _get_upper_bound(self) -> NDArray:
        return np.empty(0)

    def sample(self,
               uv: NDArray[(3, Any), float]) \
            -> Tuple[NDArray[(1, Any), float], NDArray[(Any,), bool]]:
        super().sample(uv)

        _, valid = self.camera.unproject(uv)
        constant = np.full((1,) + valid.shape, self.value)

        return constant, valid


class Cosine(Base):

    camera: Camera  # reference to camera
    k: float  # cosine exponent

    def __init__(self,
                 camera: Camera,
                 k: float) -> None:
        self.camera = camera
        self.k = k

    def _get_params(self) -> List:
        return [self.k]

    def _set_params(self, a: List) -> None:
        self.k = a[0]

    def _get_lower_bound(self) -> NDArray:
        return np.array([1.0])  # k

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf])  # k

    def sample(self,
               uv: NDArray[(3, Any), float]) \
            -> Tuple[NDArray[(1, Any), float], NDArray[(Any,), bool]]:
        super().sample(uv)

        d_c, valid = self.camera.unproject(uv)
        cos_alpha = self.camera.z.T @ d_c  # assuming d_c unit vector

        return cos_alpha ** self.k, valid


class LUT(Base):

    camera: Camera  # reference to camera
    _angles: NDArray[(Any), float]
    _values: NDArray[(Any, float)]

    def __init__(self,
                 camera: Camera,
                 step: int) -> None:
        self.camera = camera
        self._angles = np.linspace(0, np.radians(90), step)
        self._values = np.cos(self._angles) ** 2.0

    def _get_params(self) -> List:
        return self._values.tolist()[1:]

    def _set_params(self, a: List) -> None:
        self._values[1:] = np.clip(np.array(a), 0, None)

    def _get_lower_bound(self) -> NDArray:
        return np.repeat(0, self.num_params)

    def _get_upper_bound(self) -> NDArray:
        return np.repeat(1, self.num_params)

    def sample(self,
               uv: NDArray[(3, Any), float]) \
            -> Tuple[NDArray[(1, Any), float], NDArray[(Any,), bool]]:
        super().sample(uv)
        # relection angle of -w_i
        d_c, valid = self.camera.unproject(uv)
        cos_alpha = d_c[2, :]  # assuming d_c unit vector
        angle = np.arccos(cos_alpha)
        value = np.interp(angle, self._angles, self._values)
        return value, valid
