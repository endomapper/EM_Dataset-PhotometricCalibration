import re
import numpy as np
import renderers
import lights
import utils
from config_globals import OPTIMIZE_LIGHT, OPTIMIZE_VIGNETTING
import xml.etree.ElementTree as ET
from xml.dom import minidom

from typing import Any, Tuple, List, Dict
from nptyping import NDArray


def read_trajectory(file_name: str, trans_th: float = np.inf) \
        -> Tuple[List[float], Dict[float, NDArray[(4, 4), float]]]:
    ''' Read trajectory from CVS file. 
        @note Filter too fast motions
    '''
    ids = list()
    T_wc = dict()
    t_wc1 = np.eye(4)
    with open(file_name, 'r') as f:
        for line in f:
            if line.strip()[0] == '%':
                continue
            values = re.split('\s+', line)
            time = int(values[0]) + 1
            t_wc2 = np.array(values[1:13] + [0, 0, 0, 1],
                             dtype=float).reshape((4, 4))
            t_c1c2 = np.linalg.inv(t_wc1) @ t_wc2
            if np.linalg.norm(t_c1c2[0:3, 3]) < trans_th:
                ids.append(time)
                T_wc[time] = t_wc2
            t_wc1 = t_wc2
    return ids, T_wc


def save_op(file_name: str, op: NDArray[(Any,), float]) -> None:
    with open(file_name, 'w') as f:
        for o in op:
            f.write(f'{o}\n')


def load_op(file_name: str) -> NDArray[(Any,), float]:
    op = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            op += [float(line)]
    return np.array(op)


def save_config(file_name: str, cg: Dict[str, str]):
    with open(file_name, 'w') as f:
        for c in cg:
            f.write(f'{c} = {repr(cg[c])}\n')


def save_calib_xml(file_name: str, renderer: renderers.Basic):
    assert OPTIMIZE_LIGHT in ['SINGLE_NSLS', 'SINGLE_NFSLS', 'SINGLE_NSLS2D'] and \
        OPTIMIZE_VIGNETTING in ['NONE', 'COSINE'], \
        'XML export invalid for current configuration.'

    # create the file structure
    rig = ET.Element('rig')

    camera = ET.SubElement(rig, 'camera')
    camera_model = ET.SubElement(camera, 'camera_model')
    camera_model.set('name', '')
    camera_model.set('index', '0')
    camera_model.set('serialno', '0')
    camera_model.set('type', 'gamma')
    camera_model.set('version', '1.0')
    comment = ET.Comment(' Camera response model ')
    camera_model.append(comment)
    gamma = ET.SubElement(camera_model, 'gamma')
    gamma.text = f' [ {renderer.camera.gamma} ] '

    if OPTIMIZE_VIGNETTING == 'COSINE':
        vignetting = ET.SubElement(camera_model, 'vignetting')
        vignetting.text = f' [ {renderer.camera.vignetting.k:.6f} ] '

    light = ET.SubElement(rig, 'light')
    light_model = ET.SubElement(light, 'light_model')
    light_model.set('name', '')
    light_model.set('index', '0')
    light_model.set('serialno', '0')
    light_model.set('type', 'sls')
    light_model.set('version', '1.0')
    light_model.append(ET.Comment(
        ' Spot Light Source (SLS) model as in [Modrzejewski et al. (2020)] '))

    light_model.append(ET.Comment(' main intensity value '))
    sigma = ET.SubElement(light_model, 'sigma')
    sigma.text = f' {renderer.sources[0].sigma:.6f} '

    light_model.append(ET.Comment(' spread factor '))
    sigma = ET.SubElement(light_model, 'mu')
    sigma.text = f' {renderer.sources[0].mu:.6f} '

    light_model.append(ET.Comment(
        ' light centre in camera reference (3D point) '))
    P = ET.SubElement(light_model, 'P')
    P.text = f' {utils.mat2str(renderer.sources[0].P[0:3, :], 6)} '

    light_model.append(ET.Comment(
        ' principal direction in camera reference (unit 3D vector) '))
    D = ET.SubElement(light_model, 'D')
    D.text = f' {utils.mat2str(renderer.sources[0].D[0:3, :], 6)} '

    # create a new XML file with the results
    xmlstr = minidom.parseString(ET.tostring(
        rig)).childNodes[0].toprettyxml(indent="    ")
    with open(file_name, "w") as f:
        f.write(xmlstr)


def read_calib_xml(file_name: str, renderer: renderers.Basic):
    tree = ET.parse(file_name)
    rig = tree.getroot()
    for camera in rig.findall('camera'):
        camera_model = camera.find('camera_model')
        camera_type = camera_model.get('type')
        if camera_type == 'gamma':
            renderer.camera.gamma = float(camera_model.find('gamma').text.strip()[1:-1])
        else:
            raise ValueError(f'Unsupported camera type \'{camera_type}\'')
        # NOTE: only supports one camera
        break

    for i, light in enumerate(rig.findall('light')):
        light_model = light.find('light_model')
        light_type = light_model.get('type')
        if light_type == 'sls':
            sigma = float(light_model.find('sigma').text)
            mu = float(light_model.find('mu').text)
            P = utils.point(utils.str2mat(light_model.find('P').text))
            D = utils.direction(utils.str2mat(light_model.find('D').text))
            renderer.sources.append(lights.NormalizedSpotLightSource(
                sigma, mu, P, D
            ))
        else:
            raise ValueError(f'Unsupported light type \'{light_type}\'')
