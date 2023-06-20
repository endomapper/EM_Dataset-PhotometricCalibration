import os
from typing import Dict, List, Tuple
from nptyping import NDArray
import cv2
import numpy as np
from matplotlib import patches, pyplot as plt
import scipy.stats as stats
import copy
from config_globals import ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_M

import utils
import optimize
import renderers
import brdfs
import vignettings
import lights


def lightPositionFromSpecularHighlight(camera, pattern, T_wc, T_wp, img):
    T_cw = np.linalg.inv(T_wc)
    T_pw = np.linalg.inv(T_wp)
    t_wc = T_wc[0:4, 3:4]
    t_pc = T_pw @ t_wc

    # NOTE: xz_uv is the same as the principal point
    z_c = np.array([[0], [0], [1], [0]])
    z_w = T_wc @ z_c
    z_p = T_pw @ z_w
    xz_p, _ = pattern.intersect(t_pc, z_p)
    xz_w = T_wp @ xz_p
    xz_c = T_cw @ xz_w
    xz_uv, _ = camera.project(xz_c)
    # END OF NOTE

    # get point of specular higlight
    n_p = pattern.n
    n_w = T_wp @ n_p
    x0_p, _ = pattern.intersect(t_pc, n_p)
    x0_c = T_cw @ T_wp @ x0_p
    x0_uv, valid = camera.project(x0_c)

    # plot camera principal point and expected specular highlight
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='jet')
    plt.colorbar()
    plt.plot(xz_uv[0, :], xz_uv[1, :], '+k', label='Z-camera')
    if valid[0]:
        plt.plot(x0_uv[0, :], x0_uv[1, :], '+w', label='Highlight')
    plt.legend()

    # allow user to select a better specular highlight location
    x1_uv = plt.ginput(1, 0)
    plt.plot(x1_uv[0][0], x1_uv[0][1], '.k')

    # estimate light position from the "better" highlight location
    d1_c, _ = camera.unproject(np.array([[x1_uv[0][0]], [x1_uv[0][1]], [1]]))
    d1_w = T_wc @ d1_c
    x1_p, _ = pattern.intersect(t_pc, T_pw @ d1_w)
    x1_w = T_wp @ x1_p
    d2_w = 2 * (d1_w.T @ n_w) * n_w - d1_w
    camera_plane = copy.deepcopy(pattern)
    camera_plane.n = z_w
    camera_plane.d = z_w.T @ -t_wc
    xl_w, _ = camera_plane.intersect(x1_w, -d2_w)

    # plot the whole squeme of ligth-surface-camera
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot camera location and forward
    utils.drawRefSystem(ax, T_wc, 'C')
    utils.drawRefSystem(ax, T_wp, 'P')
    # plot pattern sheet
    utils.drawRectangle(ax, pattern._top_left, pattern._bottom_right, 'k')
    # plot point, w_i and w_o
    utils.drawPoint(ax, x1_w, 'X', 'k')
    utils.drawVector(ax, x1_w, -d1_w * np.linalg.norm(x1_w - t_wc),
                     label='w_o', format='--', color='k')
    utils.drawVector(ax, x1_w, -d2_w * np.linalg.norm(x1_w - xl_w),
                     label='w_i', format='--', color='k')
    # plot light
    utils.drawPoint(ax, xl_w, 'L', 'y')

    utils.axisEqual3D(ax)
    print('Distance light-camera:', np.linalg.norm(t_wc - xl_w))


def showBrdfVignettingGainSpread(op_init, op_final, renderer: renderers.Basic, gain_list):
    fig, axs = plt.subplots(2, 2)
    axs = np.ravel(axs)

    axs[0].set_title('BRDF $f_r(\omega_i, \omega_o, \mathbf{n})$')
    axs[0].set_xlabel('Angle wrt. $\omega_r$ (specular reflection)')
    axs[0].set_ylabel('Reflectance')
    if isinstance(renderer.pattern.brdf, brdfs.Diffuse):
        plotBrdfDiffuse(axs[0], renderer.pattern.brdf)
    elif isinstance(renderer.pattern.brdf, brdfs.Phong):
        plotBrdfDiffuse(axs[0], brdfs.Diffuse(),
                        linestyle='--', color='tab:blue')
        plotBrdfPhong(axs[0], renderer.pattern.brdf)
    elif isinstance(renderer.pattern.brdf, brdfs.LUT):
        plotBrdfDiffuse(axs[0], brdfs.Diffuse(),
                        linestyle='--', color='tab:blue')
        plotBrdfPhong(axs[0], brdfs.Phong(), linestyle=':', color='tab:blue')
        plotBrdfLUT(axs[0], renderer.pattern.brdf)
    else:
        raise NotImplementedError
    axs[0].legend()

    axs[1].set_title('Vignetting $V(\mathbf{x})$')
    axs[1].set_xlabel('Angle wrt. camera forward')
    axs[1].set_ylabel('Multiplier $\in [0, 1]$')
    angles = np.linspace(-renderer.camera.vignetting.camera.fov / 2,
                         renderer.camera.vignetting.camera.fov / 2, 100)
    axs[1].plot(np.degrees(angles), np.cos(angles) ** 4.0,
                '--', color='tab:gray',
                label=f'$cos^4$')
    op_backup = optimize.pack_op(renderer, gain_list)
    for op, style in [(op_init, '--'), (op_final, '-')]:
        optimize.unpack_op(op, renderer, gain_list)
        if isinstance(renderer.camera.vignetting, vignettings.Cosine):
            plotVignettingCosine(
                axs[1], renderer.camera.vignetting, angles, style=style)
        elif isinstance(renderer.camera.vignetting, vignettings.LUT):
            plotVignettingLUT(
                axs[1], renderer.camera.vignetting, angles, style=style)
        elif isinstance(renderer.camera.vignetting, vignettings.Constant):
            plotVignettingConstant(
                axs[1], renderer.camera.vignetting, angles, style=style)
        else:
            raise NotImplementedError
    optimize.unpack_op(op_backup, renderer, gain_list)
    axs[1].legend()

    axs[2].set_title('Auto-gain $g_t$')
    axs[2].set_xlabel('# frame')
    axs[2].set_ylabel('Relative gain')
    axs[2].plot(gain_list)

    axs[3].set_title('Light spread $\mu(\mathbf{x})$')
    axs[3].set_xlabel('Angle w.r.t. principal')
    axs[3].set_ylabel('Multiplier R(x)')
    angles = np.linspace(-renderer.camera.vignetting.camera.fov / 2,
                         renderer.camera.vignetting.camera.fov / 2, 100)
    axs[3].plot(np.degrees(angles), np.cos(angles),
                '--', color='tab:gray', label='cos')
    op_backup = optimize.pack_op(renderer, gain_list)
    for op, style in [(op_init, '--'), (op_final, '-')]:
        optimize.unpack_op(op, renderer, gain_list)
        axs[3].set_prop_cycle(None)
        for i, source in enumerate(renderer.sources):
            if isinstance(source, lights.SpotLightSource):
                plotSpotLightSource(axs[3], source, i, angles, style=style)
            else:
                raise NotImplementedError
    optimize.unpack_op(op_backup, renderer, gain_list)
    axs[3].legend()


def plotBrdfDiffuse(ax, brdf: brdfs.Diffuse, **kwargs):
    z_d = np.array([[0.0], [0.0], [1.0], [0.0]])
    value = brdf.sample(w_i=z_d, w_o=z_d, n=z_d)
    ax.plot([-180, 180], [value[0, 0], value[0, 0]], label='diffuse', **kwargs)


def _plotBrdfSample(brdf, angles):
    w_i = np.repeat(np.array([[0], [0], [1], [0]]), len(angles), axis=1)
    w_o = np.c_[np.sin(angles), np.zeros_like(angles),
                np.cos(angles), np.zeros_like(angles)].T
    values = brdf.sample(w_i, w_o, w_i)
    return values


def plotBrdfPhong(ax, brdf: brdfs.Phong, **kwargs):
    angles = np.linspace(0, np.pi, 180)
    values = _plotBrdfSample(brdf, angles)
    plot_degrees = np.concatenate([-np.degrees(angles)[::-1],
                                   np.degrees(angles)])
    plot_phong = np.concatenate([values[0, ::-1], values[0]])
    ax.plot(plot_degrees, plot_phong, label='Phong', **kwargs)


def plotBrdfLUT(ax, brdf: brdfs.LUT):
    scatter_degrees = np.concatenate([-np.degrees(brdf._angles)[::-1],
                                      np.degrees(brdf._angles)])
    scatter_lut = np.concatenate([brdf._values[::-1],
                                  brdf._values])
    ax.scatter(scatter_degrees, scatter_lut, marker='x')
    angles = np.linspace(0, np.pi, 180)
    values = _plotBrdfSample(brdf, angles)
    plot_degrees = np.concatenate([-np.degrees(angles)[::-1],
                                   np.degrees(angles)])
    plot_lut = np.concatenate([values[0, ::-1], values[0]])
    ax.plot(plot_degrees, plot_lut, '-',
            color='tab:blue', label=f'LUT ({brdf.num_params})')


def plotVignettingConstant(ax, vignetting: vignettings.Constant, angles, style='-', color='tab:blue'):
    ax.plot(np.degrees(angles), np.cos(angles) ** 0 * vignetting.value,
            style, color=color, label=f'const')


def plotVignettingCosine(ax, vignetting: vignettings.Cosine, angles, style='-', color='tab:blue'):
    ax.plot(np.degrees(angles), np.cos(angles) ** vignetting.k,
            style, color=color, label=f'$cos^{{{vignetting.k:.2f}}}$')


def plotVignettingLUT(ax, vignetting: vignettings.LUT, angles, style='-', color='tab:blue'):
    ax.plot(np.degrees(angles),
            np.interp(np.abs(angles), vignetting._angles, vignetting._values),
            style, color=color, label=f'LUT ({vignetting.num_params})')


def plotSpotLightSource(ax, source, i, angles, style='-'):
    values = np.exp(-source.mu * (1 - np.cos(angles)))
    ax.plot(np.degrees(angles), values, style,
            label=f'light {i} ($\\mu = {source.mu:.2f}$)')


def compareInputAndRender(img_gray, render, valid):
    error = (img_gray - np.mean(render, axis=2)[:, :, np.newaxis])
    error[np.logical_not(valid)] = 0.0

    plt.figure()
    plt.title('I_hat')
    plt.imshow(render)
    plt.figure()
    plt.title('I - I_hat')
    plt.imshow(error * 255, vmin=-25, vmax=25, cmap='seismic')
    plt.colorbar()


def compareCurrentVsDiffuseBrdf(renderer: renderers.Basic, T_wc, T_wp, gain):
    current_render, _ = renderer.full(T_wc, T_wp, gain)
    current_brdf = renderer.pattern.brdf
    renderer.pattern.brdf = brdfs.Diffuse()
    render_diffuse, _ = renderer.full(T_wc, T_wp, gain)
    renderer.pattern.brdf = current_brdf
    plt.figure()
    plt.title('I_diffuse')
    plt.imshow(render_diffuse)
    plt.figure()
    plt.title('I_hat - I_diffuse')
    difference = np.mean(current_render * 255, axis=2)[:, :, np.newaxis] - \
        np.mean(render_diffuse * 255, axis=2)[:, :, np.newaxis]
    min_max = max(np.abs(np.min(difference)), np.abs(np.max(difference)))
    plt.imshow(difference, vmin=-min_max, vmax=min_max, cmap='seismic')
    plt.colorbar()


def plotPatternSample(pattern, **kwargs):
    ''' Plot the sample points on the pattern '''
    plt.figure()
    m2px = pattern._mask.shape[1] / pattern._size[0]
    import patterns
    if type(pattern) is patterns.Vicalib:
        # Add the patch to the Axes
        rect = patches.Rectangle(pattern._top_left * m2px,
                                pattern.width * m2px, pattern.height * m2px,
                                linewidth=1, edgecolor='k', facecolor='none')
        plt.gca().add_patch(rect)
        plot_x_p = (pattern.sample(**kwargs) + pattern.large_rad) * m2px
    else:
        plot_x_p = (pattern.sample(**kwargs) + pattern.spacing) * m2px
    plt.imshow(pattern._mask)
    plt.scatter(plot_x_p[0], plot_x_p[1], c='r', s=10)
    plt.axis('equal')


def plotCameraSample(camera, uv):
    ''' Plot the sample points on the pattern '''
    plt.figure()
    plt.imshow(camera.mask)
    plt.scatter(uv[0], uv[1], c='r', s=10)
    plt.axis('equal')


def sourcesOnEndoscope(axs: plt.Axes,
                       sources: List[lights.Base],
                       section_path: str,
                       center_px: Tuple[float, float],
                       diameter_px: float,
                       diameter_m: float):
    img = cv2.imread(section_path)
    axs.imshow(img)
    px2uv = 1. / diameter_px * img.shape[0]
    m2px = diameter_px / diameter_m
    axs.plot(center_px[0] * px2uv, center_px[1] * px2uv, 'ks', label='camera')
    for i, source in enumerate(sources):
        axs.plot((center_px[0] - source.d_xy[0] * m2px) * px2uv,
                 (center_px[1] + source.d_xy[1] * m2px) * px2uv, '.',
                 label=f'light{i}')


def show2DVignettingAndLightSpread(op_final, renderer: renderers.Basic):
    fig, axs = getSquaredGrid(
        len(renderer.sources) + 3, orientation='portrait')
    uv = renderer.camera.sample()
    vignetting, valid = renderer.camera.vignetting.sample(uv)
    vignetting[:, np.logical_not(valid)] = 0.0
    resolution = (renderer.camera.resolution[1, 0],
                  renderer.camera.resolution[0, 0])
    vig_show = axs[0].imshow(vignetting.reshape(resolution))
    plt.colorbar(vig_show, ax=axs[0])
    axs[0].plot(renderer.camera.Cx, renderer.camera.Cy, 'xk', label='Z')
    axs[0].set_title('Vignetting')
    axs[0].legend()

    ray, mask = renderer.camera.unproject(uv)
    T_wc = np.eye(4)
    x_w = ray
    x_w[3, :] = 1
    for i, ax, source in zip(range(len(axs) - 1), axs[1:-1], renderer.sources):
        l, wi_w = source.sample(T_wc, x_w)
        l[:, np.logical_not(mask)] = 0.0
        l_show = ax.imshow(l.reshape(resolution))
        plt.colorbar(l_show, ax=ax)
        ax.plot(renderer.camera.Cx, renderer.camera.Cy, 'xk', label='Z')
        D_uv, _ = renderer.camera.project(T_wc[:, 3:4] + source.D)
        ax.plot(D_uv[0, 0], D_uv[1, 0], '.k', label='D')
        deg = np.degrees(np.arccos(source.D[2, 0]))
        ax.set_title(f'Light {i} @ ${deg:.1f}^\circ$')
        ax.legend()

    MARGIN = 0.15
    axs[-2].margins(MARGIN)
    circle = plt.Circle((0, 0), ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_M / 2.0 * 1000,
                        color='k', fill=False)
    axs[-2].add_patch(circle)
    axs[-2].text(0, 0.5, 'Left', transform=axs[-2].transAxes,
                 rotation=90, horizontalalignment='left', verticalalignment='center')
    axs[-2].text(1, 0.5, 'Right', transform=axs[-2].transAxes,
                 rotation=-90, horizontalalignment='right', verticalalignment='center')
    axs[-2].text(0.5, 0, 'Down', transform=axs[-2].transAxes,
                 rotation=0, horizontalalignment='center', verticalalignment='bottom')
    axs[-2].text(0.5, 1, 'Up', transform=axs[-2].transAxes,
                 rotation=0, horizontalalignment='center', verticalalignment='top')
    axs[-2].plot(0, 0, 'xk', label='C')
    for i, source in enumerate(renderer.sources):
        axs[-2].plot(source.P[0, 0] * 1000, source.P[1, 0] * 1000,
                     '.', label=f'L{i}')
    axs[-2].set_aspect('equal', 'box')
    axs[-2].invert_yaxis()
    axs[-2].legend()
    axs[-2].set_title('Displacement XY (mm)')
    axs[-2].set_xlabel('X')
    axs[-2].set_ylabel('Y')

    axs[-1].margins(MARGIN)
    axs[-1].hlines(y=0.0,
                   xmin=-ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_M / 2.0 * 1000,
                   xmax=ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_M / 2.0 * 1000,
                   colors='k', linestyles='-')
    axs[-1].text(0, 0.5, 'Left', transform=axs[-1].transAxes,
                 rotation=90, horizontalalignment='left', verticalalignment='center')
    axs[-1].text(1, 0.5, 'Right', transform=axs[-1].transAxes,
                 rotation=-90, horizontalalignment='right', verticalalignment='center')
    axs[-1].text(0.5, 0, 'Back', transform=axs[-1].transAxes,
                 rotation=0, horizontalalignment='center', verticalalignment='bottom')
    axs[-1].text(0.5, 1, 'Front', transform=axs[-1].transAxes,
                 rotation=0, horizontalalignment='center', verticalalignment='top')
    axs[-1].plot(0, 0, 'xk', label='C')
    for i, source in enumerate(renderer.sources):
        axs[-1].plot(source.P[0, 0] * 1000, source.P[2, 0] * 1000,
                     '.', label=f'L{i}')
    axs[-1].set_aspect('equal', 'box')
    xlim = max([abs(lim) for lim in axs[-1].get_xlim()])
    ylim = max([abs(lim) for lim in axs[-1].get_ylim()])
    lim = max(xlim, ylim)
    axs[-1].set_xlim(left=-lim, right=lim)
    axs[-1].set_ylim(bottom=-lim, top=lim)
    axs[-1].legend()
    axs[-1].set_title('Displacement XZ (mm)')
    axs[-1].set_xlabel('X')
    axs[-1].set_ylabel('Z')


def getSquaredGrid(n: int, title: str = '', orientation: str = 'landscape'):
    if orientation == 'landscape':
        fig, axs = plt.subplots(
            np.floor(np.sqrt(n)).astype(int),
            np.ceil(n / np.floor(np.sqrt(n))).astype(int))
    elif orientation == 'portrait':
        fig, axs = plt.subplots(
            np.ceil(np.sqrt(n)).astype(int),
            np.ceil(n / np.ceil(np.sqrt(n))).astype(int))
    else:
        raise ValueError(f'orientation \'{orientation}\' unknown.')
    axs = axs.ravel()
    for ax in axs[n:]:
        ax.remove()
    fig.suptitle(title)
    return fig, axs[:n]


def tukeysRuleFilter(x):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    mask = np.logical_and(x >= q25 - 1.5 * iqr, x <= q75 + 1.5 * iqr)
    return x[mask]


def plotGaussian(ax, residuals):
    mu = np.mean(tukeysRuleFilter(residuals))
    sigma = np.std(tukeysRuleFilter(residuals))
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma),
            c='orangered', label='Fitted Gaussian')


def checkIfThreeLightsAreWorthIt(
        renderer: renderers.Basic,
        frame_ids: List[int],
        frame_poses: Dict[int, NDArray[(4, 4), float]],
        T_wp,
        data: str,
        mode: str = 'static'):

    T_pw = np.linalg.inv(T_wp)

    def wz_dst(id):
        T_wc = frame_poses[id]
        return abs(T_wc[2, 3]) * 100

    def cz_dst(id):
        T_wc = frame_poses[id]
        x, _ = renderer.pattern.intersect(
            T_pw @ T_wc[:, 3:4], T_pw @ T_wc[:, 2:3])
        return np.linalg.norm(x - T_wc[:, 3:4]) * 100

    def th_ngl(id):
        T_wc = frame_poses[id]
        return np.degrees(np.arccos(T_wp[:, 2] @ -T_wc[:, 2]))

    def get_cross_location_normal(id):
        T_wc = frame_poses[id]
        T_cw = np.linalg.inv(T_wc)
        uv_cross = []
        for source in renderer.sources:
            t_cl = np.array([[source.d_xy[0]], [source.d_xy[1]], [0], [1]])
            t_wl = T_wc @ t_cl
            # asssuming T_wp ~ I
            x_nl = np.array([[t_wl[0, 0]], [t_wl[1, 0]], [0], [1]])
            uv, _ = renderer.camera.project(T_cw @ x_nl)
            uv_cross.append(uv)
        return uv_cross

    def get_cross_location_forward(id):
        T_wc = frame_poses[id]
        T_cw = np.linalg.inv(T_wc)
        uv_cross = []
        for source in renderer.sources:
            t_cl = np.array([[source.d_xy[0]], [source.d_xy[1]], [0], [1]])
            t_wl = T_wc @ t_cl
            d_wl = T_wc @ source.D
            x_nl, _ = renderer.pattern.intersect(
                T_pw @ t_wl, T_pw @ d_wl)  # asssuming T_wp ~ I
            uv, _ = renderer.camera.project(T_cw @ x_nl)
            uv_cross.append(uv)
        return uv_cross

    def showFrame(ax, id):
        min_img = cv2.imread(os.path.join(data, f'{id:06d}.png'))
        # min_img = cv2.bilateralFilter(min_img, 29, 25, 100)
        im = ax.imshow(min_img[:, :, 0], cmap='tab20', vmin=100)
        objs = [im]
        ax.set_prop_cycle(None)
        for cross in get_cross_location_normal(id):
            cr1, = ax.plot(cross[0], cross[1], 'wx',
                           markeredgewidth=2.5, markeredgecolor='k')
            cr2, = ax.plot(cross[0], cross[1], 'x')
            objs += [cr1, cr2]
        ax.set_prop_cycle(None)
        for cross in get_cross_location_forward(id):
            pt1, = ax.plot(cross[0], cross[1], 'w.',
                           markeredgewidth=2.5, markeredgecolor='k')
            pt2, = ax.plot(cross[0], cross[1], '.')
            objs += [pt1, pt2]
        return objs

    if mode == 'static':
        fig, axs = plt.subplots(1, 3)
        for ax, name, key in zip(axs, ['Lowest (cm)', 'Closest (cm)', 'Theta (deg)'], [wz_dst, cz_dst, th_ngl]):
            min_id = min(frame_poses, key=key)
            ax.set_title(f'{name} {key(min_id):.2f} @ {min_id:06d}')
            showFrame(ax, id)
        plt.show()
    elif mode == 'video':
        # NOTE: range for HCULB_03020 (up and down)
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        legend_elements = [
            plt.Line2D([], [], color='black', linestyle='None',
                       marker='x', label='vertical'),
            plt.Line2D([], [], color='black', linestyle='None',
                       marker='.', label='forward')
        ]
        # ax.set_title(f'{id:06d} @ N:{wz_dst(id):.2f} cm @ Z:{cz_dst(id):.2f} cm @ {th_ngl(id):.2f} deg')

        def animate(id):
            ax.clear()
            ax.set_axis_off()
            frame = showFrame(ax, id)
            text_elements = [
                plt.Line2D([], [], linestyle='None', label=f'ID: {id:06d}'),
                plt.Line2D([], [], linestyle='None',
                           label=f'N: {wz_dst(id):.2f} cm'),
                plt.Line2D([], [], linestyle='None',
                           label=f'Z: {cz_dst(id):.2f} cm'),
                plt.Line2D([], [], linestyle='None',
                           label=f'TH: {th_ngl(id):.2f}$^\circ$'),
            ]
            # text = ax.legend(handles=text_elements, loc='upper left', handlelength=0, handletextpad=0)
            legend = ax.legend(handles=text_elements + legend_elements,
                               loc='upper left', handlelength=1, handletextpad=0.4)
            plt.savefig(f'anim/{id:06d}.png', bbox_inches='tight', dpi=300)
            return frame + [legend]
        f_range = [id for id in range(5780, 6185) if id in frame_poses]
        anim = animation.FuncAnimation(fig, animate, frames=f_range,
                                       interval=20, blit=True, repeat=False)
        # writer = animation.writers['ffmpeg'](fps=30)
        # anim.save('demo.mp4', writer=writer, dpi=300)
        plt.show()
    else:
        raise ValueError(f'Invalid debug mode: {mode}')
