import cv2
import numpy as np
from xml.dom import minidom
import fqs

from base import Camera as Base
import vignettings

from config_globals import OPTIMIZE_VIGNETTING

from typing import Any, List, Tuple
from nptyping import NDArray

class Factory(object):
    def fromXML(xml: str, mask: str) -> Base:
        file = minidom.parse(xml)
        type = file.getElementsByTagName('camera_model')[0].getAttribute('type')
        if type == 'calibu_fu_fv_u0_v0':
            return Pinhole(xml, mask)
        elif type == 'calibu_fu_fv_u0_v0_k1_k2':
            return Poly2(xml, mask)
        elif type == 'calibu_fu_fv_u0_v0_kb4':
            return Fisheye(xml, mask)
        elif type == 'ocamcalib_cx_cy_a0_a2_a3_a4_e_f_g':
            return Scaramuzza(xml, mask)
        else:
            raise ValueError(
                f'Invalid camera model type in XML file: \'{type}\'')

class Pinhole(Base):

    z: NDArray[(4, 1), float]  # camera forward

    ''' Intrinsic parameters '''
    fx: float  # focal length x-axis (px)
    fy: float  # focal length y-axis (px)
    Cx: float  # principal point x-axis (px)
    Cy: float  # principal point y-axis (px)

    ''' Aditional parameters '''
    width: int  # px
    height: int  # px
    fov: float  # rad
    mask: NDArray[(Any, Any), bool]  # valid image mask

    ''' Photometric parameters '''
    vignetting: vignettings.Base
    gamma: float  # response param

    @property
    def resolution(self) -> NDArray[(2), int]:
        return np.array([[self.width],
                         [self.height]], dtype=int)

    @property
    def fxy(self) -> NDArray[(2), float]:
        return np.array([[self.fx],
                         [self.fy]])

    @property
    def Cxy(self) -> NDArray[(2), float]:
        return np.array([[self.Cx],
                         [self.Cy]])

    @property
    def K (self) -> NDArray[(3, 3), float]:
        return np.array([[self.fx, 0, self.Cx],
                         [0, self.fy, self.Cy],
                         [0, 0, 1]])

    def __init__(self, calib_file: str, mask_file: str):
        self.z = np.array([[0], [0], [1], [0]])
        Pinhole._load_params(self, calib_file, mask_file)
        if OPTIMIZE_VIGNETTING == 'COSINE':
            self.vignetting = vignettings.Cosine(self, 2.0)
        elif OPTIMIZE_VIGNETTING == 'LUT':
            self.vignetting = vignettings.LUT(self, 15)
        elif OPTIMIZE_VIGNETTING == 'NONE':
            self.vignetting = vignettings.Constant(self, 1.0)
        else:
            raise ValueError(
                f'Invalid OPTIMIZE_VIGNETTING: \'{OPTIMIZE_VIGNETTING}\'')
        self.gamma = 2.2
        self.fov = np.radians(74.46357)

    def _load_params(self, calib_file: str, mask_file: str):
        ''' Load camera parameters '''
        # parse an xml file by name
        file = minidom.parse(calib_file)

        self.width = int(
            file.getElementsByTagName('width')[0].firstChild.data)
        self.height = int(
            file.getElementsByTagName('height')[0].firstChild.data)

        self.mask = cv2.imread(mask_file)[:, :, 0].astype(bool)

        params = file.getElementsByTagName('params')[0].firstChild.data
        params = params[2:-2].split('; ')

        self.fx = float(params[0])
        self.fy = float(params[1])
        self.Cx = float(params[2])
        self.Cy = float(params[3])

    def project(self, x_c: NDArray[(4, Any), float], **kwargs) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(Any,), bool]]:
        ''' Porject points (camera frame) to image space (pixels) '''
        assert x_c.shape[0] == 4, '`x_c` must be homogeneous coordinates'
        assert np.all(x_c[3, :] == 1), '`x_c` must be point coordinates'

        uv = self.fxy * x_c[0:2] / x_c[2:3] + self.Cxy
        uv = np.vstack((uv, np.ones(uv.shape[1])))
        valid = self.is_valid(uv)
        return uv, valid

    def is_valid(self, uv: NDArray[(3, Any), bool]) \
            -> NDArray[(Any,), bool]:
        ''' Check if pixel coordinates are valid '''
        assert uv.shape[0] == 3, '`uv` must be homogeneous coordinates'
        assert np.all(uv[2, :] == 1), '`uv` must be pixel coordinates'
        valid = np.logical_and(
            np.all(np.floor(uv[0:2, :]) >= np.zeros((2, 1)), axis=0),
            np.all(np.ceil(uv[0:2, :]) < self.resolution, axis=0))

        def floor(x, y): return np.clip(np.floor(x), 0, y - 1).astype(int)
        def ceil(x, y): return np.clip(np.ceil(x), 0, y - 1).astype(int)
        valid[valid] = np.logical_and(
            valid[valid], self.mask[
                floor(uv[1, valid], self.mask.shape[0]),
                floor(uv[0, valid], self.mask.shape[1])])
        valid[valid] = np.logical_and(
            valid[valid], self.mask[
                ceil(uv[1, valid], self.mask.shape[0]),
                floor(uv[0, valid], self.mask.shape[1])])
        valid[valid] = np.logical_and(
            valid[valid], self.mask[
                floor(uv[1, valid], self.mask.shape[0]),
                ceil(uv[0, valid], self.mask.shape[1])])
        valid[valid] = np.logical_and(
            valid[valid], self.mask[
                ceil(uv[1, valid], self.mask.shape[0]),
                ceil(uv[0, valid], self.mask.shape[1])])

        return valid

    def unproject(self, uv: NDArray[(3, Any), float], **kwargs) \
            -> Tuple[NDArray[(4, Any), float], NDArray[(Any,), bool]]:
        ''' Unproject pixels to rays (camera frame) '''
        assert uv.shape[0] == 3, '`uv` must be homogeneous coordinates'
        assert np.all(uv[2, :] == 1), '`uv` must be pixel coordinates'

        valid = self.is_valid(uv)
        d_c = (uv[0:2] - self.Cxy) / self.fxy
        d_c = np.vstack((d_c, np.ones(d_c.shape[1]), np.zeros(d_c.shape[1])))
        d_c /= np.linalg.norm(d_c, axis=0)
        
        return d_c, valid

    def response(self, L: NDArray[(Any, Any), float]) \
            -> NDArray[(Any, Any), float]:
        ''' Apply response function to input L values '''
        return np.sign(L) * np.abs(L) ** (1.0 / self.gamma)

    def inv_response(self, L: NDArray[(Any, Any), float]) \
            -> NDArray[(Any, Any), float]:
        ''' Apply inverse response function to input L values '''
        return np.sign(L) * np.abs(L) ** (self.gamma)

    def sample(self) -> NDArray[(3, Any), float]:
        ''' Returns all the UV coordinates on the image '''
        resolution = list(self.resolution.flatten())[::-1]
        uv = np.moveaxis(np.indices(resolution), 0, -1)
        uv = uv[:, :, ::-1]  # list of all the uv in the image
        uv = uv.reshape(-1, 2).astype(np.float32).T
        uv += 0.5  # middle of the pixel
        uv = np.r_[uv, np.ones((1, uv.shape[1]))]
        return uv

class Poly2(Pinhole):

    k1: float  # distortion coefficients [k1, k2]
    k2: float  # such that x' = x * (1 + k1*r² + k2*r⁴)

    def __init__(self, calib_file: str, mask_file: str):
        super(). __init__(calib_file, mask_file)
        Poly2._load_params(self, calib_file, mask_file)
        self.fov = np.radians(136.0)

    def _load_params(self, calib_file, mask_file):
        file = minidom.parse(calib_file)
        params = file.getElementsByTagName('params')[0].firstChild.data
        params = params[2:-2].split('; ')
        self.k1 = float(params[4])
        self.k2 = float(params[5])

    def factor(self, rad):
        ''' Implementation borrowed from Calibu '''
        r2 = rad * rad
        r4 = r2 * r2
        return 1.0 + self.k1 * r2 + self.k2 * r4
    
    def factor_inv(self, r):
        ''' Implementation borrowed from Calibu '''
        # Use Newton's method to solve (fixed number of iterations)
        # (for explanation, see notes in beginning of camera_models_crtp.h)
        ru = np.copy(r)
        for i in range(5):
            # Common sub-expressions of d, d2
            ru2 = ru * ru
            ru4 = ru2 * ru2
            pol = self.k1 * ru2 + self.k2 * ru4 + 1
            pol2 = 2 * ru2 * (self.k1 + 2 * self.k2 * ru2)
            pol3 = pol + pol2

            # 1st derivative
            d = (ru * (pol) - r)  *  2 * pol3
            # 2nd derivative
            d2 = (4 * ru * (ru * pol - r) *
                    (3 * self.k1 + 10 * self.k2 * ru2 ) +
                    2 * pol3 * pol3)
            # Delta update
            delta = d / d2
            ru -= delta

        # Return the undistortion factor
        return ru / r
    
    def project(self, x_c: NDArray[(4, Any), float], **kwargs) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(Any,), bool]]:
        ''' Implementation borrowed from Calibu '''
        uv = x_c[0:2] / x_c[2:3]
        fac = self.factor(np.linalg.norm(uv[0:2, :], axis=0, keepdims=True))
        uv[0:2, :] *= fac
        uv = self.fxy * uv + self.Cxy
        uv = np.vstack((uv, np.ones(uv.shape[1])))
        valid = self.is_valid(uv)
        return uv, valid
    
    def unproject(self, uv: NDArray[(3, Any), float], **kwargs) \
            -> Tuple[NDArray[(4, Any), float], NDArray[(Any,), bool]]:
        ''' Implementation borrowed from Calibu '''
        valid = self.is_valid(uv)
        d_c = (uv[0:2, :] - self.Cxy) / self.fxy
        fac_inv = self.factor_inv(np.linalg.norm(d_c[0:2, :], axis=0, keepdims=True))
        d_c[0:2, :] *= fac_inv
        d_c = np.vstack((d_c, np.ones(d_c.shape[1]), np.zeros(d_c.shape[1])))
        return d_c, valid

class Fisheye(Pinhole):

    ''' Intrinsic parameters '''
    D: List[float]  # distortion coefficients [k1, k2, k3, k4]

    def __init__(self, calib_file: str, mask_file: str):
        super(). __init__(calib_file, mask_file)
        Fisheye._load_params(self, calib_file, mask_file)
        self.fov = np.radians(136.0)

    def _load_params(self, calib_file: str, mask_file: str):
        ''' Load camera parameters '''
        # parse an xml file by name
        file = minidom.parse(calib_file)
        params = file.getElementsByTagName('params')[0].firstChild.data
        params = params[2:-2].split('; ')
        self.D = [float(d) for d in params[4:8]]

    def _d(self, theta: NDArray[(1, Any), float]) \
            -> NDArray[(1, Any), float]:
        ''' Calculate the value of the d(theta) polynomial '''
        theta2 = theta * theta
        theta3 = theta * theta2
        theta5 = theta3 * theta2
        theta7 = theta5 * theta2
        theta9 = theta7 * theta2
        return theta + self.D[0] * theta3 + self.D[1] * theta5 \
                     + self.D[2] * theta7 + self.D[3] * theta9

    def project(self, x_c: NDArray[(4, Any), float], **kwargs) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(Any,), bool]]:
        ''' Porject points (camera frame) to image space (pixels) '''
        assert x_c.shape[0] == 4, '`x_c` must be homogeneous coordinates'
        assert np.all(x_c[3, :] == 1), '`x_c` must be point coordinates'

        x2_plus_y2 = x_c[0] * x_c[0] + x_c[1] * x_c[1]
        theta = np.arctan2(np.sqrt(x2_plus_y2), x_c[2])
        psi = np.arctan2(x_c[1], x_c[0])
        r = self._d(theta)
        u = self.fx * r * np.cos(psi) + self.Cx
        v = self.fy * r * np.sin(psi) + self.Cy
        uv = np.vstack((u, v, np.ones(v.shape[0])))
        # NOTE: cannot project OOB pixels
        valid = np.abs(theta) < self.fov / 2.0
        valid = np.logical_and(valid, self.is_valid(uv))
        return uv, valid

    def unproject(self, uv: NDArray[(3, Any), float], **kwargs) \
            -> Tuple[NDArray[(4, Any), float], NDArray[(Any,), bool]]:
        ''' Unproject pixels to rays (camera frame) '''
        assert uv.shape[0] == 3, '`uv` must be homogeneous coordinates'
        assert np.all(uv[2, :] == 1), '`uv` must be pixel coordinates'

        precision = kwargs['precision'] if 'precision' in kwargs else 1e-6

        uv = uv[0:2, :] / uv[2:3, :]
        pw = (uv - self.Cxy) / self.fxy
        theta_d = np.linalg.norm(pw, axis=0)
        # NOTE: cannot unproject OOB pixels
        valid = np.abs(theta_d) < self._d(self.fov / 2.0)

        mask_ = np.logical_and(valid, theta_d > 1e-8)
        theta_ = np.copy(theta_d)

        for _ in range(10):
            theta = theta_[mask_]
            theta2 = theta * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta
            theta8 = theta4 * theta4

            k0 = self.D[0] * theta2
            k1 = self.D[1] * theta4
            k2 = self.D[2] * theta6
            k3 = self.D[3] * theta8
            theta_fix = (theta * (1 + k0 + k1 + k2 + k3) - theta_d[mask_]) / \
                        (1 + 3 * k0 + 5 * k1 + 7 * k2 + 9 * k3)

            theta_[mask_] = theta - theta_fix
            # NOTE: this mask_ is used as a loop-break, so that:
            #       `if (abs(theta_fix) < precision) --> break`
            mask_[mask_] = abs(theta_fix) >= precision

        scale = np.tan(theta_) / theta_d

        d_c = np.vstack(
            (pw * scale, np.ones(scale.shape), np.zeros(scale.shape)))
        d_c /= np.linalg.norm(d_c, axis=0)

        valid[valid] = np.logical_and(valid[valid],
                                      self.mask[(uv[1, valid]).astype(
                                          int), (uv[0, valid]).astype(int)]
                                      )

        return d_c, valid
    
class Scaramuzza(Pinhole):

    a0: float  # distortion coefficients [a0, a2, a3, a4]
    a2: float
    a3: float
    a4: float
    e: float
    f: float
    g: float

    def __init__(self, calib_file: str, mask_file: str):
        # super().__init__(calib_file, mask_file)
        Scaramuzza._load_params(self, calib_file, mask_file)
        self.mask = cv2.imread(mask_file)[:, :, 0].astype(bool)
        self.fov = np.radians(136.0)

    def _load_params(self, calib_file, mask_file):
        self.width = 1350
        self.height = 1080
        self.cx = 679.544839263292
        self.cy = 543.975887548343
        self.a0 = 769.243600037458
        self.a2 = -0.000812770624150226
        self.a3 = 6.25674244578925e-07
        self.a4 = -1.19662182144280e-09
        self.e = 0.999986882249990
        self.f = 0.00288273829525059
        self.g = -0.00296316513429569
    
    def project(self, x_c: NDArray[(4, Any), float], **kwargs) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(Any,), bool]]:
        ''' Implementation borrowed from Calibu '''
        xx = np.copy(x_c)
        xx = xx[0:3, :]

        ind0 = np.any(xx[:2] == 0, axis=0)
        xx[:2, ind0] = kwargs['eps'] if 'eps' in kwargs else 1e-9

        m = xx[2, :] / np.linalg.norm(xx[:2, :], axis=0)

        coeffs = np.zeros((len(m), 5))
        coeffs[:, 0] = self.a0
        coeffs[:, 1] = -m
        coeffs[:, 2] = self.a2
        coeffs[:, 3] = self.a3
        coeffs[:, 4] = self.a4
        roots = fqs.quartic_roots(coeffs[:, ::-1])
        roots = roots[np.isreal(roots)]
        rho = np.real(roots[roots > 0])

        u = xx[0, :] / np.linalg.norm(xx[:2, :], axis=0) * rho
        v = xx[1, :] / np.linalg.norm(xx[:2, :], axis=0) * rho
        uv_ = np.stack([u, v])

        A = np.array([[self.e, self.f],
                    [self.g, 1.0]])
        uv = A @ uv_ + [[self.cx], [self.cy]]
        uv = np.vstack((uv, np.ones(uv.shape[1])))
        valid = self.is_valid(uv)
        return uv, valid
    
    def unproject(self, uv: NDArray[(3, Any), float], **kwargs) \
            -> Tuple[NDArray[(4, Any), float], NDArray[(Any,), bool]]:
        ''' Implementation borrowed from Calibu '''
        valid = self.is_valid(uv)
        uv = uv[0:2, :].T
        A = np.array([[self.e, self.f],
                  [self.g, 1.0]])
        uv_ = np.linalg.inv(A) @ (uv.reshape(-1, 2).T - [[self.cx], [self.cy]])

        def f(self, rho):
            return self.a0 + self.a2 * rho ** 2 + self.a3 * rho ** 3 + self.a4 * rho ** 4

        d_c = np.vstack([uv_, f(self, np.linalg.norm(uv_, axis=0, keepdims=True))])
        d_c /= d_c[2:3, :]
        d_c = d_c.reshape(3, -1)
        d_c = np.vstack((d_c, np.zeros(d_c.shape[1])))
        d_c /= np.linalg.norm(d_c, axis=0)
        return d_c, valid