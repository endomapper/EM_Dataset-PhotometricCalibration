import numpy as np
import scipy.optimize

import renderers
from config_globals import OPTIMIZE_GAIN

from typing import Any, List
from nptyping import NDArray


def pack_op(renderer: renderers.Basic,
            gain: List[float]):
    # Optimization parameteres
    op = []
    op += [renderer.camera.vignetting.params]
    op += [renderer.pattern.brdf.params]
    for i in range(len(renderer.sources)):
        op += [renderer.sources[i].params]
    if OPTIMIZE_GAIN == 'ALL':
        op += [gain[:]]
    elif OPTIMIZE_GAIN == 'EXCLUDE_FIRST':
        op += [gain[1:]]
    else:
        raise ValueError(f'Invalid OPTIMIZE_GAIN: \'{OPTIMIZE_GAIN}\'')
    return np.concatenate(op)


def unpack_op(op: NDArray[(Any, ), float],
              renderer: renderers.Basic,
              gain: List[float]):
    inf = 0
    sup = renderer.camera.vignetting.num_params
    renderer.camera.vignetting.params = op[inf:sup].tolist()
    inf = sup
    sup = inf + renderer.pattern.brdf.num_params
    renderer.pattern.brdf.params = op[inf:sup].tolist()
    for i in range(len(renderer.sources)):
        inf = sup
        sup = inf + renderer.sources[i].num_params
        renderer.sources[i].params = op[inf:sup].tolist()
    inf = sup
    if OPTIMIZE_GAIN == 'ALL':
        gain[:] = op[inf:]
    elif OPTIMIZE_GAIN == 'EXCLUDE_FIRST':
        gain[1:] = op[inf:]
    else:
        raise ValueError(f'Invalid OPTIMIZE_GAIN: \'{OPTIMIZE_GAIN}\'')


def bounds(renderer: renderers.Basic,
           gain: List[float]):
    lower_bound = renderer.camera.vignetting.lower_bound
    upper_bound = renderer.camera.vignetting.upper_bound
    lower_bound = np.concatenate(
        [lower_bound, renderer.pattern.brdf.lower_bound])
    upper_bound = np.concatenate(
        [upper_bound, renderer.pattern.brdf.upper_bound])
    for i in range(len(renderer.sources)):
        lower_bound = np.concatenate(
            [lower_bound, renderer.sources[i].lower_bound])
        upper_bound = np.concatenate(
            [upper_bound, renderer.sources[i].upper_bound])
    lower_bound = np.concatenate([lower_bound, np.repeat(
        1e-6, len(gain) + (0 if OPTIMIZE_GAIN == 'ALL' else -1))])
    upper_bound = np.concatenate([upper_bound, np.repeat(
        np.inf, len(gain) + (0 if OPTIMIZE_GAIN == 'ALL' else -1))])
    return (lower_bound, upper_bound)


def jac_sparsity(I_gt: List[float],
                 op: NDArray[(Any, ), float]):
    n_frames = len(I_gt)
    m = sum(len(I_gt) for I_gt in I_gt)  # num. residuals
    n = len(op)  # num. variables
    from scipy.sparse import lil_matrix
    sparsity = lil_matrix((m, n), dtype=int)
    sparsity[:, 0:n-n_frames] = 1
    m_ = 0 if OPTIMIZE_GAIN == 'ALL' else len(I_gt[0])
    for i in range(0, n_frames):
        num_residuals_in_frame = len(I_gt[i])
        sparsity[m_:m_+num_residuals_in_frame, -n_frames + i] = 1
        m_ += num_residuals_in_frame
    return sparsity


def fun(op: NDArray[(Any, ), float],
        x_w: List[NDArray[(4, Any), float]],
        x_valid: List[NDArray[(Any, ), float]],
        T_wc: List[NDArray[(4, 4), float]],
        T_wp: NDArray[(4, 4), float],
        I_gt: List[NDArray[(Any, 1), float]],
        renderer: renderers.Basic,
        gain: List[float],
        unpack_op=unpack_op):
    n_frames = len(gain)
    gain_ = np.copy(gain)
    unpack_op(op, renderer, gain_)
    residuals = []
    for i in range(n_frames):
        I_hat, _ = renderer.x_w(
            x_w[i][:, x_valid[i]], T_wc[i], T_wp, gain_[i])
        I_hat = np.mean(I_hat, axis=0)[:, np.newaxis]
        residuals.append(I_hat - I_gt[i])
    return np.squeeze(np.concatenate(residuals))


def fun_debug(op: NDArray[(Any, ), float],
              x_w: List[NDArray[(4, Any), float]],
              x_valid: List[NDArray[(Any, ), float]],
              T_wc: List[NDArray[(4, 4), float]],
              T_wp: NDArray[(4, 4), float],
              I_gt: List[NDArray[(Any, 1), float]],
              renderer: renderers.Basic,
              gain: List[float],
              unpack_op=unpack_op):
    n_frames = len(gain)
    gain_ = np.copy(gain)
    unpack_op(op, renderer, gain_)
    residuals = []
    angle_wrt_forward = []
    angle_wrt_normal = []
    for i in range(n_frames):
        rays_w = x_w[i][:, x_valid[i]] - T_wc[i][:, 3:4]
        rays_w = rays_w / np.linalg.norm(rays_w, axis=0)[np.newaxis, :]
        T_cw = np.linalg.inv(T_wc[i])
        rays_c = T_cw @ rays_w
        angle_wrt_forward.append(np.arccos(rays_c.T @ renderer.camera.z))
        T_pw = np.linalg.inv(T_wp)
        rays_p = T_pw @ rays_w
        angle_wrt_normal.append(np.arccos(-rays_p.T @ renderer.pattern.n))
        I_hat, _ = renderer.x_w(
            x_w[i][:, x_valid[i]], T_wc[i], T_wp, gain_[i])
        I_hat = np.mean(I_hat, axis=0)[:, np.newaxis]
        residuals.append(I_hat - I_gt[i])
    return np.squeeze(np.concatenate(residuals)), \
        np.squeeze(np.concatenate(angle_wrt_forward)), \
        np.squeeze(np.concatenate(angle_wrt_normal))


def pack_op_test(_: renderers.Basic,
                 gain: List[float]):
    return gain


def unpack_op_test(op, _: renderers.Basic,
                   gain: List[float]):
    gain[:] = op


def eval_test(x_w: List[NDArray[(4, Any), float]],
              x_valid: List[NDArray[(Any, ), float]],
              T_wc: List[NDArray[(4, 4), float]],
              T_wp: NDArray[(4, 4), float],
              I_gt: List[NDArray[(Any, 1), float]],
              renderer: renderers.Basic,
              gain: List[float]):
    gain_ = np.copy(gain)
    op_init = pack_op_test(None, gain_)
    result = scipy.optimize.least_squares(
        fun,
        op_init,
        method='trf',
        xtol=1e-15, ftol=1e-15, gtol=1e-15,
        x_scale='jac',
        loss='huber',
        verbose=0,
        args=(x_w, x_valid, T_wc,
              T_wp, I_gt, renderer, gain_, unpack_op_test)
    )
    op_final = result.x
    unpack_op_test(op_final, None, gain_)
    residuals = fun(op_final, x_w, x_valid,
                    T_wc, T_wp, I_gt, renderer, gain_, unpack_op_test)
    return residuals, gain_
