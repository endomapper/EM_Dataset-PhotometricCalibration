import numpy as np

import utils

from typing import Any, List
from nptyping import NDArray


class Base(utils.Optimizable):

    def sample(self,
               w_i: NDArray[(4, Any), float],
               w_o: NDArray[(4, Any), float],
               n: NDArray[(4, Any), float]) \
            -> NDArray[(3, Any), float]:
        ''' Sample BRDF for two given directions '''
        assert w_i.shape[0] == 4, '`w_i` must be homogeneous coordinates'
        assert w_o.shape[0] == 4, '`w_o` must be homogeneous coordinates'
        assert n.shape[0] == 4, '`n` must be homogeneous coordinates'
        assert np.all(
            w_i[3, :] == 0), '`w_i` must be direction coordinates'
        assert np.all(
            w_o[3, :] == 0), '`w_o` must be direction coordinates'
        assert np.all(
            n[3, :] == 0), '`n` must be direction coordinates'
        assert np.allclose(np.linalg.norm(w_i, axis=0),
                           1), '`w_i` must be unitary directions'
        assert np.allclose(np.linalg.norm(w_o, axis=0),
                           1), '`w_o` must be unitary directions'
        assert np.allclose(np.linalg.norm(n, axis=0),
                           1), '`n` must be unitary directions'
        pass


class Diffuse(Base):

    def __init__(self) -> None:
        pass

    def sample(self,
               w_i: NDArray[(4, Any), float],
               w_o: NDArray[(4, Any), float],
               n: NDArray[(4, Any), float]) \
            -> NDArray[(3, Any), float]:
        super().sample(w_i, w_o, n)
        return np.full((3, w_i.shape[1]), 1/np.pi)

    def _get_params(self) -> List:
        return []

    def _set_params(self, a: List) -> None:
        pass

    def _get_lower_bound(self) -> NDArray:
        return np.empty(0)

    def _get_upper_bound(self) -> NDArray:
        return np.empty(0)


class Phong(Base):

    _ks: float  # Phong's specular coefficient
    _n: float  # Phong's specular exponent

    def __init__(self) -> None:
        self._ks = 0.05
        self._n = 1.0

    def _get_params(self) -> List:
        return [self._ks, self._n]

    def _set_params(self, a: List) -> None:
        self._ks = a[0]
        self._n = a[1]

    def _get_lower_bound(self) -> NDArray:
        return np.array([0, 1])  # _ks, _n

    def _get_upper_bound(self) -> NDArray:
        return np.array([1, np.inf])  # _ks, _n

    def sample(self,
               w_i: NDArray[(4, Any), float],
               w_o: NDArray[(4, Any), float],
               n: NDArray[(4, Any), float]) \
            -> NDArray[(3, Any), float]:
        super().sample(w_i, w_o, n)
        w_r = 2 * np.sum(n * w_i, axis=0)[np.newaxis, :] * n - w_i
        cosine = np.clip(np.sum(w_o * w_r, axis=0), 0, None)[np.newaxis, :]
        value = (1 - self._ks) / np.pi + self._ks * cosine ** self._n
        return np.full((3, w_i.shape[1]), value)


class LUT(Base):

    _angles: NDArray[(Any), float]
    _values: NDArray[(Any, float)]

    def __init__(self, step: int) -> None:
        self._angles = np.linspace(0, np.radians(160), step)
        self._values = np.full(step, 1/np.pi)

    def _get_params(self) -> List:
        return self._values.tolist()[1:]

    def _set_params(self, a: List) -> None:
        self._values[1:] = np.array(a)

    def _get_lower_bound(self) -> NDArray:
        return np.repeat(0, self.num_params)

    def _get_upper_bound(self) -> NDArray:
        return np.repeat(1, self.num_params)

    def sample(self,
               w_i: NDArray[(4, Any), float],
               w_o: NDArray[(4, Any), float],
               n: NDArray[(4, Any), float]) \
            -> NDArray[(3, Any), float]:
        super().sample(w_i, w_o, n)
        # relection angle of -w_i
        w_r = 2 * np.sum(n * w_i, axis=0)[np.newaxis, :] * n - w_i
        cosine = np.sum(w_o * w_r, axis=0)[np.newaxis, :]
        angle = np.arccos(cosine)
        value = np.interp(angle, self._angles, self._values)
        return np.full((3, w_i.shape[1]), value)
