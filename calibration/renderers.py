import numpy as np

from cameras import Fisheye
from patterns import Vicalib
import brdfs
import lights

from typing import Any, List, Tuple
from nptyping import NDArray


class Basic:

    camera: Fisheye
    sources: List[lights.Base]
    pattern: Vicalib

    def __init__(self,
                 camera: Fisheye,
                 sources: List[lights.Base],
                 pattern: Vicalib) -> None:
        self.camera = camera
        self.sources = sources
        self.pattern = pattern

    def x_w(self,
            x_w: NDArray[(4, Any), float],
            T_wc: NDArray[(4, 4), float],
            T_wp: NDArray[(4, 4), float],
            gain: float) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(Any,), bool]]:
        ''' Render image at world points '''
        T_pw = np.linalg.inv(T_wp)
        T_cw = np.linalg.inv(T_wc)

        # get the light that reaches the surface points `x_w`
        # NOTE: takes into account both emission and distance attenuation

        # surface properties
        x_p = T_pw @ x_w
        n_p = np.repeat(self.pattern.n, x_p.shape[1], axis=1)
        albedo, valid_albedo = self.pattern.albedo(x_p)

        wo = T_wc[0:4, 3:4] - x_w
        wo = wo / np.linalg.norm(wo, axis=0)

        # FIXME: First time you check this implementation actually
        #        supports multiple light return values, remove this assert
        L_o = 0
        for source in self.sources:
            li, wi_w = source.sample(T_wc, x_w)
            cosine = np.sum((T_wp @ n_p) * wi_w, axis=0)[np.newaxis, :]
            brdf = self.pattern.brdf.sample(T_pw @ wi_w, T_pw @ wo, n_p)
            # render equation
            L_o += li * albedo * brdf * cosine

        uv, valid_uv = self.camera.project(T_cw @ x_w)
        vignetting = np.zeros((1, uv.shape[1]))
        vignetting[:, valid_uv], valid_vignetting = \
            self.camera.vignetting.sample(uv[:, valid_uv])

        bgr = self.camera.response(L_o * vignetting * gain)
        valid = np.logical_and(valid_albedo, valid_uv)
        valid[valid_uv] = np.logical_and(valid[valid_uv], valid_vignetting)

        return bgr, valid

    def uv(self,
           uv: NDArray[(3, Any), float],
           T_wc: NDArray[(4, 4), float],
           T_wp: NDArray[(4, 4), float],
           gain: float) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(Any,), bool]]:
        ''' Render image at pixels '''
        d_c, d_c_valid = self.camera.unproject(uv)
        d_w = T_wc @ d_c
        d_w_valid = d_c_valid
        x_w, x_w_valid = \
            self.pattern.intersect(T_wc[0:4, 3:4], d_w[:, d_w_valid])

        bgr_, valid_bgr = self.x_w(x_w[:, x_w_valid], T_wc, T_wp, gain)
        bgr = np.zeros((3, uv.shape[1]))
        valid = d_w_valid
        valid[valid] = x_w_valid
        bgr[:, valid] = bgr_
        valid[valid] = valid_bgr

        return bgr, valid

    def full(self,
             T_wc: NDArray[(4, 4), float],
             T_wp: NDArray[(4, 4), float],
             gain: float) \
            -> Tuple[NDArray[(Any, Any, 3), float],
                     NDArray[(Any, Any, 1), bool]]:
        ''' Render full image '''
        # Sample all pixels
        uv = self.camera.sample()

        bgr, valid = self.uv(uv, T_wc, T_wp, gain)

        resolution = (self.camera.resolution[1, 0],
                      self.camera.resolution[0, 0],
                      3)
        render = np.zeros(resolution)
        render[uv[1, valid].astype(int),
               uv[0, valid].astype(int), :] = bgr[:, valid].T
        resolution = (self.camera.resolution[1, 0],
                      self.camera.resolution[0, 0],
                      1)
        mask = np.zeros(resolution, dtype=bool)
        mask[uv[1, valid].astype(int), uv[0, valid].astype(int), :] = True

        return render, mask


class Unity(Basic):

    def __init__(self,
                 camera: Fisheye,
                 sources: List[lights.Base],
                 brdf: brdfs.Base) -> None:
        self.camera = camera
        self.sources = sources

    def full(self,
             depth: NDArray[(Any, Any, 1), float],
             normal: NDArray[(Any, Any, 3), float],
             albedo: NDArray[(Any, Any, 3), float],
             gain: float) \
            -> Tuple[NDArray[(Any, Any, 3), float],
                     NDArray[(Any, Any, 1), bool]]:
        ''' Render full image '''
        pass
