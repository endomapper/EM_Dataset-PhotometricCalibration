from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, List, Tuple
from nptyping import NDArray


class Optimizable(object):

    def __init__(self) -> None:
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        return len(self.params)

    @property
    def params(self) -> List:
        return self._get_params()

    @property
    def lower_bound(self) -> NDArray:
        bound = self._get_lower_bound()
        assert len(bound) == self.num_params, \
            f'Invalid lower bound dims. Expected {self.nun_params}, {len(bound)} given.'
        return bound

    @property
    def upper_bound(self) -> NDArray:
        bound = self._get_upper_bound()
        assert len(bound) == self.num_params, \
            f'Invalid upper bound dims. Expected {self.nun_params}, {len(bound)} given.'
        return bound

    @params.setter
    def params(self, a: List) -> None:
        assert len(a) == self.num_params, \
            f'Expected {self.num_params} parameters, {len(a)} given.'
        self._set_params(a)

    def _set_params(self, a: List) -> None:
        raise NotImplementedError

    def _get_params(self) -> List:
        raise NotImplementedError

    def _get_lower_bound(self) -> NDArray:
        raise NotImplementedError

    def _get_upper_bound(self) -> NDArray:
        raise NotImplementedError


def interpolate(img: NDArray[(Any, Any, Any), Any],
                uv: NDArray[(3, Any), float]) \
        -> Tuple[NDArray[(Any, Any), Any], NDArray[(Any,), bool]]:
    ''' Bicubic interpolation '''
    uv = uv[0:2, :] / uv[2:3, :]

    a_max = [[img.shape[1] - 1], [img.shape[0] - 1]]
    prev = (np.clip(np.floor(uv), 0, a_max)).astype(int)
    next = (np.clip(np.floor(uv) + 1, 0, a_max)).astype(int)

    a00 = ((next[0] - uv[0]) * (next[1] - uv[1]))[:, np.newaxis]
    a01 = ((uv[0] - prev[0]) * (next[1] - uv[1]))[:, np.newaxis]
    a10 = ((next[0] - uv[0]) * (uv[1] - prev[1]))[:, np.newaxis]
    a11 = ((uv[0] - prev[0]) * (uv[1] - prev[1]))[:, np.newaxis]

    values = \
        img[prev[1], prev[0]] * a00 + \
        img[prev[1], next[0]] * a01 + \
        img[next[1], prev[0]] * a10 + \
        img[next[1], next[0]] * a11

    valid = np.logical_and(
        np.all(uv >= 0.0, axis=0),
        np.all(uv < a_max, axis=0)
    )
    values[np.logical_not(valid), :] = 0
    values = values.astype(img.dtype)
    return values, valid


def axisEqual3D(ax):
    ''' Set axis equal in 3d plot '''
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))()
                        for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    ax.set_box_aspect((4, 4, 4))


def drawPoint(ax,
              x: NDArray[(4, 1), float],
              label: str,
              color: str = None):
    ''' Plot a single point with given text label '''
    ax.plot(x[0, 0], x[1, 0], x[2, 0], '.', color=color)
    ax.text(x[0, 0],
            x[1, 0],
            x[2, 0],
            label, zorder=1, color='k')


def drawVector(ax,
               o: NDArray[(4, 1), float],
               d: NDArray[(4, 1), float],
               label='',
               format: str = '-',
               color: str = None):
    ''' Plot a single vector from origin with given text label '''
    ax.plot([o[0, 0], o[0, 0] + d[0, 0]],
            [o[1, 0], o[1, 0] + d[1, 0]],
            [o[2, 0], o[2, 0] + d[2, 0]], format, color=color)
    if label != '':
        x = o + d / 2
        ax.text(x[0, 0], x[1, 0], x[2, 0],
                label, (d[0, 0], d[1, 0], d[2, 0]),
                zorder=1, color='k')


def drawRefSystem(ax,
                  T_wc: NDArray[(4, 4), float],
                  label: str,
                  s: float = 0.01):
    ax.text(T_wc[0, 3],
            T_wc[1, 3],
            T_wc[2, 3],
            label, zorder=1, color='k')
    drawVector(ax, T_wc[:, 3:4], T_wc[:, 0:1] * s, color='r')
    drawVector(ax, T_wc[:, 3:4], T_wc[:, 1:2] * s, color='g')
    drawVector(ax, T_wc[:, 3:4], T_wc[:, 2:3] * s, color='b')


def drawRectangle(ax,
                  top_left: NDArray[(2, 1), float],
                  bottom_right: NDArray[(2, 1), float],
                  color=None):
    ax.plot([top_left[0, 0], top_left[0, 0],
             bottom_right[0, 0], bottom_right[0, 0],
             top_left[0, 0]],
            [top_left[1, 0], bottom_right[1, 0],
             bottom_right[1, 0], top_left[1, 0],
             top_left[1, 0]],
            [0, 0, 0, 0, 0], '-', color=color)


def sphere2cartesian(theta: float,
                     phi: float,
                     r: float = 1.0) \
        -> NDArray[(4, 1), float]:
    return np.array([[r * np.cos(phi) * np.sin(theta)],
                     [r * np.sin(phi) * np.sin(theta)],
                     [r * np.cos(theta)],
                     [0]])


def cartesian2sphere(xyzw: NDArray[(4, 1), float]) \
        -> Tuple[float, float, float]:
    r = np.linalg.norm(xyzw)
    theta = np.arctan2(np.linalg.norm(xyzw[0:2, 0]), xyzw[2, 0])
    phi = np.arctan2(xyzw[1, 0], xyzw[0, 0])
    return theta, phi, r


def dl2pose(theta, phi, t_cl=[[0], [0], [0]]):
    zc = [0, 0, 1]
    zl = sphere2cartesian(theta, phi)[:3, 0]
    zl = zl / np.linalg.norm(zl)
    yl = np.cross(zl, zc)
    yl = yl / np.linalg.norm(yl)
    xl = np.cross(yl, zl)
    xl = xl / np.linalg.norm(xl)
    T_cl = np.eye(4)
    T_cl[0:3, 0] = xl
    T_cl[0:3, 1] = yl
    T_cl[0:3, 2] = zl
    T_cl[0:3, 3:4] = t_cl
    return T_cl


def savefig_pdf_multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf', bbox_inches='tight', dpi=600)
    pp.close()


def mat2str(mat: NDArray[Any, Any], decimals: int = None):
    if decimals is not None:
        mat = np.round(mat, decimals)
    return '[ ' + '; '.join([', '.join(b) for b in mat.astype(str).tolist()]) + ' ]'

def direction(vect3: NDArray[(3, 1), Any]):
    return np.r_[(vect3, [[0]])]

def point(vect3: NDArray[(3, 1), Any]):
    return np.r_[(vect3, [[1]])]

def str2mat(string: str, dtype = float):
    arr = [col.split(',') for col in string.strip()[1:-1].split(';')]
    return np.array(arr, dtype=dtype)