import numpy as np
from scipy.spatial.transform import Rotation as R

import utils

from typing import Any, List, Tuple
from nptyping import NDArray


class Base(utils.Optimizable):

    def sample(self,
               T_wc: NDArray[(4, 1), float],
               x_w: NDArray[(4, Any), float]) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(4, Any), float]]:
        ''' Sample light _comming_ to a point in space'''
        assert T_wc.shape == (4, 4), \
            f'`T_wc` attached camera pose must be (4, 4), but {T_wc.shape} encountered.'
        assert x_w.shape[0] == 4, '`x_w` must be homogeneous coordinates'
        assert np.allclose(x_w[3, :], 1), '`x_w` must be point coordinates'
        pass

###############################################################################
##                              COMPLETE CLASS                               ##
###############################################################################


class SpotLightSource(Base):
    '''Spot Light Source (SLS). [Modrzejewski20]
    - Main intensity value: σ_o
    - Light center: P (coordinates XYZ)
    - Normalized principal direction: L(x, P) = (x - P) / ||x - P||
    - Inverse square law: S(x, P) = 1/d², where d = ||x - P||
    - Directional D spread function: R(μ, D, x, P) = e^(-μ(1 - D·L))
    - σ_SLS(x, P) = σ_o · R(μ, D, x, P) · S(x, P) · L(x, P)
    '''

    sigma: float  # main intensity value
    mu: float  # spread factor
    P: NDArray[(4, 1), float]  # light centre in camera reference
    D: NDArray[(4, 1), float]  # principal direction in camera reference

    def __init__(self,
                 sigma: float = 1.0,
                 mu: float = 0.0,
                 P: NDArray[(4, 1), float] = np.array(
                     [[0.], [0.], [0.], [1.]]),
                 D: NDArray[(4, 1), float] = np.array(
                     [[0.], [0.], [1.], [0.]])) -> None:
        assert D.shape == (4, 1), '`D` must be homogeneous direction'
        self.sigma = sigma
        self.mu = mu
        self.P = P
        self.D = D

    def sample(self,
               T_wc: NDArray[(4, 1), float],
               x_w: NDArray[(4, Any), float]) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(4, Any), float]]:
        super().sample(T_wc, x_w)

        T_cw = np.linalg.inv(T_wc)
        x_c = T_cw @ x_w

        vP2x = x_c - self.P
        d = np.linalg.norm(vP2x, axis=0)[np.newaxis, :]
        L_x = vP2x / d
        S_x = 1 / (d * d)
        R_x = np.exp(-self.mu * (1 - self.D.T @ L_x))

        sigma_SLS = self.sigma * R_x * S_x * L_x

        # return value and direction separately
        value = np.linalg.norm(sigma_SLS, axis=0)[np.newaxis, :]
        w_i = T_wc @ -L_x
        return value, w_i

    def _get_params(self) -> List:
        params = [self.sigma]
        params += [self.mu]
        params += self.P.flatten().tolist()[0:3]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]
        self.mu = a[1]
        self.P = np.array(a[2:5] + [1, ]).reshape(4, 1)
        self.D = utils.sphere2cartesian(a[5], a[6])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # sigma
                         0,        # mu
                         -np.inf,  # P_x
                         -np.inf,  # P_y
                         -np.inf,  # P_z
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # sigma
                         np.inf,   # mu
                         np.inf,   # P_x
                         np.inf,   # P_y
                         np.inf,   # P_z
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])

    @property
    def T_cl(self) -> NDArray[(4, 4), float]:
        z = np.array([[0], [0], [1], [0]])
        rotvec = np.cross(z[:3, :].T, self.D[:3, :].T).T
        rotvec /= np.linalg.norm(rotvec)
        rotvec *= np.arccos(np.dot(z.T, self.D)[0, 0])
        rot = R.from_rotvec(rotvec.T)
        rotmat = rot.as_matrix()
        T_cl = np.eye(4)
        T_cl[:3, :3] = rotmat
        T_cl[:, 3] = self.P
        return T_cl

###############################################################################
##                          OPTIMIZABLE VARIATIONS                           ##
###############################################################################


class NormalizedSpotLightSource(SpotLightSource):
    ''' SLS with normalized radiance '''

    def _get_params(self) -> List:
        params = [self.mu]
        params += self.P.flatten().tolist()[0:3]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.mu = a[0]
        self.P = np.array(a[1:4] + [1, ]).reshape(4, 1)
        self.D = utils.sphere2cartesian(a[4], a[5])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # mu
                         -np.inf,  # P_x
                         -np.inf,  # P_y
                         -np.inf,  # P_z
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # mu
                         np.inf,   # P_x
                         np.inf,   # P_y
                         np.inf,   # P_z
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class SpotLightSource2D(SpotLightSource):
    ''' SLS with light center in optical plane '''

    def _get_params(self) -> List:
        params = [self.sigma]
        params += [self.mu]
        params += self.P.flatten().tolist()[0:2]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]
        self.mu = a[1]
        self.P = np.array(a[2:4] + [0, 1]).reshape(4, 1)
        self.D = utils.sphere2cartesian(a[4], a[5])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # sigma
                         0,        # mu
                         -np.inf,  # P_x
                         -np.inf,  # P_y
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # sigma
                         np.inf,   # mu
                         np.inf,   # P_x
                         np.inf,   # P_y
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class NormalizedSpotLightSource2D(SpotLightSource):
    ''' SLS with light center in optical plane and normalized radiance '''

    def _get_params(self) -> List:
        params = [self.mu]
        params += self.P.flatten().tolist()[0:2]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.mu = a[0]
        self.P = np.array(a[1:3] + [0, 1]).reshape(4, 1)
        self.D = utils.sphere2cartesian(a[3], a[4])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # mu
                         -np.inf,  # P_x
                         -np.inf,  # P_y
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # mu
                         np.inf,   # P_x
                         np.inf,   # P_y
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class FixedSpotLightSource(SpotLightSource):
    ''' Fixed Spot Light Source (FSLS) '''

    def _get_params(self) -> List:
        params = [self.sigma]
        params += [self.mu]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]
        self.mu = a[1]
        self.D = utils.sphere2cartesian(a[2], a[3])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # sigma
                         0,        # mu
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # sigma
                         np.inf,   # mu
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class NormalizedFixedSpotLightSource(SpotLightSource):
    ''' Fixed Spot Light Source (FSLS) with normalized radiance '''

    def _get_params(self) -> List:
        params = [self.mu]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.mu = a[0]
        self.D = utils.sphere2cartesian(a[1], a[2])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # mu
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # mu
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class ZFixedSpotLightSource(SpotLightSource):
    ''' Fixed Spot Light Source (FSLS) with Z principal direction '''

    def _get_params(self) -> List:
        params = [self.sigma]
        params += [self.mu]
        return params

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]
        self.mu = a[1]

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # sigma
                         0,        # mu
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # sigma
                         np.inf,   # mu
                         ])


class NormalizedZFixedSpotLightSource(SpotLightSource):
    ''' Fixed Spot Light Source (FSLS) with Z principal direction and normalized radiance '''

    def _get_params(self) -> List:
        params = [self.mu]
        return params

    def _set_params(self, a: List) -> None:
        self.mu = a[0]

    def _get_lower_bound(self) -> NDArray:
        return np.array([0])  # mu

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf])  # mu


class FixedPointLightSource(SpotLightSource):
    ''' Fixed Point Light Source (FPLS) '''

    def _get_params(self) -> List:
        return [self.sigma]

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]

    def _get_lower_bound(self) -> NDArray:
        return np.array([0])  # sigma

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf])  # sigma


class NormalizedFixedPointLightSource(SpotLightSource):
    ''' Fixed Point Light Source (FPLS) with normalized radiance '''

    def _get_params(self) -> List:
        return []

    def _set_params(self, a: List) -> None:
        pass

    def _get_lower_bound(self) -> NDArray:
        return np.empty(0)

    def _get_upper_bound(self) -> NDArray:
        return np.empty(0)
