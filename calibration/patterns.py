import os
import cv2
import numpy as np
from xml.dom import minidom

import brdfs
import utils
from config_globals import OPTIMIZE_BRDF

from typing import Any, Tuple
from nptyping import NDArray

class Factory:

    def fromXML(params_file: str):
        file = minidom.parse(params_file)
        type = file.getElementsByTagName('preset')[0].getAttribute('type')
        if type == 'photodepth_c_r_s':
            # checkerboard
            return Checkerboard(params_file)
        elif type == 'photodepth_c_r_s_lr_sr':
            # vicalib
            return Vicalib(params_file)

class Vicalib:
    ''' Vicalib pattern 
        @note T_wp reference frame is on lop left circle
    '''

    width: float  # sheet width (m) --longer side--
    height: float  # sheet height (m)

    columns: int  # number of columns
    rows: int  # number of rows

    spacing: float  # grid spacing (m) between centers
    large_rad: float  # large radius (m)
    small_rad: float  # small radius (m)

    ''' @note following are in T_wp reference '''
    T_wp: NDArray[(4, 4), float]  # reference frame
    n: NDArray[(4, 1), float]  # normal of the plane
    d: float  # distance to the origin (m)

    def __init__(self, params_file: str):
        self.n = np.array([[0], [0], [1], [0]])
        self.d = 0
        if OPTIMIZE_BRDF == 'DIFFUSE':
            self.brdf = brdfs.Diffuse()
        elif OPTIMIZE_BRDF == 'PHONG':
            self.brdf = brdfs.Phong()
        elif OPTIMIZE_BRDF == 'LUT':
            self.brdf = brdfs.LUT(15)
        else:
            raise ValueError(f'Invalid OPTIMIZE_BRDF: \'{OPTIMIZE_BRDF}\'')
        self._load_params(params_file)
        self.T_wp = np.array([[1, 0,  0, 0],
                              [0, 1,  0, 0],
                              [0, 0, -1, 0],
                              [0, 0,  0, 1]])

    def _load_params(self, params_file: str):
        ''' Load camera parameters '''
        # parse an xml file by name
        file = minidom.parse(params_file)
        SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
        file_png = os.path.join(SCRIPT_PATH, 'vicalib.png')
        self._mask = cv2.imread(file_png, cv2.IMREAD_COLOR) / 255.0

        self.width = int(
            file.getElementsByTagName('width')[0].firstChild.data) / 1000.0
        self.height = int(
            file.getElementsByTagName('height')[0].firstChild.data) / 1000.0

        params = file.getElementsByTagName('params')[0].firstChild.data
        params = params[2:-2].split('; ')

        self.columns = int(params[0])
        self.rows = int(params[1])
        self.spacing = float(params[2]) / 1000.0
        self.large_rad = float(params[3]) / 1000.0
        self.small_rad = float(params[4]) / 1000.0

        self._size = (np.array([[self.columns], [self.rows]]) - 1) \
            * self.spacing + 2 * self.large_rad
        _sheet_size = [[self.width], [self.height]]
        _sheet_margin = (_sheet_size - self._size) / 2.0
        self._top_left = -(_sheet_margin + self.large_rad)
        self._bottom_right = self._top_left + _sheet_size
        self._m2px = [[self._mask.shape[1]], [
            self._mask.shape[0]]] / self._size

    def intersect(self, o: NDArray[(4, 1), float], d: NDArray[(4, Any), float]) \
            -> Tuple[NDArray[(4, Any), float], NDArray[(Any,), bool]]:
        '''
        Intersect multiple rays from a single origin.

        @param o: common origin
        @param d: directions of the rays
        @return: intersection points, valid mask
        '''
        # plane and ray are parallel and do not meet
        valid = np.abs(self.n.T @ d) >= 1e-5
        alpha = -(self.d + np.dot(o.T, self.n)) / np.dot(self.n.T, d)
        # looking at the skybox
        valid = np.logical_and(valid, alpha >= 0)
        x = o + d * alpha
        return x, valid.reshape(-1)

    def albedo(self, x_p: NDArray[(4, Any), float]) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(Any,), bool]]:
        ''' Get albedo of a point in the pattern '''
        ''' @note margin in meters '''
        assert x_p.shape[0] == 4, '`x_p` must be homogeneous coordinates'
        assert np.all(x_p[3, :] == 1), '`x_p` must be point coordinates'
        assert np.all(np.abs(x_p[2, :]) <=
                      1e-10), '`points must be on Z=0 plane'

        x_uv = x_p[0:2, :] + self.large_rad
        x_uv = x_uv * self._m2px

        x_uv_homo = np.r_[x_uv, np.ones((1, x_uv.shape[1]))]

        albedo = np.ones((3, x_p.shape[1]), dtype=float)
        _albedo, _valid = utils.interpolate(self._mask, x_uv_homo)
        albedo[:, _valid] = _albedo[_valid, :].T

        valid = np.logical_and(
            np.all(x_p[0:2, :] >= self._top_left, axis=0),
            np.all(x_p[0:2, :] < self._bottom_right, axis=0)
        )

        return albedo, valid

    def sample(self, margin: int = 0, **kwargs) \
            -> NDArray[(4, Any), float]:
        ''' Get sample points on the pattern '''
        if type(margin) == tuple or type(margin) == list:
            if len(margin) == 2:
                margin = np.array(
                    [[margin[1], margin[1]], [margin[0], margin[0]]])
            elif len(margin) == 3:
                margin = np.array(
                    [[margin[1], margin[1]], [margin[0], margin[2]]])
            elif len(margin) == 4:
                margin = np.array(
                    [[margin[3], margin[1]], [margin[0], margin[2]]])
            else:
                raise ValueError
        elif type(margin) == int:
            margin = np.array([[margin, margin], [margin, margin]])
        else:
            raise ValueError

        tl = np.floor(self._top_left / self.spacing +
                      margin[:, 0:1]) * self.spacing
        br = np.floor(self._bottom_right /
                      self.spacing - margin[:, 1:2]) * self.spacing
        x_p = self.spacing / 2 + \
            np.mgrid[tl[0]:br[0]:self.spacing,
                     tl[1]:br[1]:self.spacing]
        x_p = x_p.reshape(2, -1)
        x_p = np.r_[x_p, np.zeros((1, x_p.shape[1])),
                    np.ones((1, x_p.shape[1]))]
        return x_p

class Checkerboard(Vicalib):
    ''' Checkerboard pattern 
        @note T_wp reference frame is on lop left circle
    '''

    def __init__(self, params_file: str):
        super().__init__(params_file)
        self.n = np.array([[0], [0], [-1], [0]])
        self.d = 0
        self._load_params(params_file)
        self.T_wp = np.eye(4)

    def _load_params(self, params_file: str):
        ''' Load camera parameters '''
        # parse an xml file by name
        file = minidom.parse(params_file)
        SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
        file_png = os.path.join(SCRIPT_PATH, 'checkerboard.png')
        self._mask = cv2.imread(file_png, cv2.IMREAD_COLOR) / 255.0

        self.width = int(
            file.getElementsByTagName('width')[0].firstChild.data) / 1000.0
        self.height = int(
            file.getElementsByTagName('height')[0].firstChild.data) / 1000.0

        params = file.getElementsByTagName('params')[0].firstChild.data
        params = params[2:-2].split('; ')

        self.columns = int(params[0])
        self.rows = int(params[1])
        self.spacing = float(params[2]) / 1000.0

        self._size = np.array([[self.columns], [self.rows]]) * self.spacing
        _sheet_size = [[self.width], [self.height]]
        _sheet_margin = (_sheet_size - self._size) / 2.0
        self._top_left = -(_sheet_margin + self.spacing)
        self._bottom_right = self._top_left + _sheet_size
        self._m2px = [[self._mask.shape[1]], [
            self._mask.shape[0]]] / self._size

    def albedo(self, x_p: NDArray[(4, Any), float]) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(Any,), bool]]:
        ''' Get albedo of a point in the pattern '''
        ''' @note margin in meters '''
        assert x_p.shape[0] == 4, '`x_p` must be homogeneous coordinates'
        assert np.all(x_p[3, :] == 1), '`x_p` must be point coordinates'
        assert np.all(np.abs(x_p[2, :]) <=
                      1e-10), '`points must be on Z=0 plane'

        x_uv = x_p[0:2, :] + self.spacing
        x_uv = x_uv * self._m2px

        x_uv_homo = np.r_[x_uv, np.ones((1, x_uv.shape[1]))]

        albedo = np.ones((3, x_p.shape[1]), dtype=float)
        _albedo, _valid = utils.interpolate(self._mask, x_uv_homo)
        albedo[:, _valid] = _albedo[_valid, :].T

        valid = np.logical_and(
            np.all(x_p[0:2, :] >= self._top_left, axis=0),
            np.all(x_p[0:2, :] < self._bottom_right, axis=0)
        )

        return albedo, valid

    def sample(self, margin: int = 0, **kwargs) \
            -> NDArray[(4, Any), float]:
        ''' Get sample points on the pattern '''
        ''' @note ignore margin '''

        # TODO poner varios puntos en cada caudrado vacío

        tl = self._top_left + self.spacing
        br = self._bottom_right + self.spacing
        x_p = self.spacing / 2 + \
            np.mgrid[tl[0]:br[0]:self.spacing,
                     tl[1]:br[1]:self.spacing]
        x_p = x_p.reshape(2, -1)
        white_r1 = [False, True] * (self.rows // 2) + \
                ([False] if self.rows % 2 == 1 else [])
        white_r2 = [True, False] * (self.rows // 2) + \
                ([True] if self.rows % 2 == 1 else [])
        white = (white_r1 + white_r2) * (self.columns // 2) + \
                (white_r1 if self.columns % 2 == 1 else [])
        x_p = x_p[:, white]
        x_p = x_p - self.spacing
        x_p = np.r_[x_p, np.zeros((1, x_p.shape[1])),
                    np.ones((1, x_p.shape[1]))]

        if 'repeat' in kwargs:
            x_p = [x_p]
            for rp_x, rp_y in kwargs['repeat']:
                x_p.append(x_p[0] + [[rp_x * self.spacing], 
                                     [rp_y * self.spacing], [0], [0]])
            x_p = np.hstack(x_p)

        return x_p
