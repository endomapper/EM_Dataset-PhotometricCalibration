from typing import Any, Tuple
from nptyping import NDArray

class Camera(object):

    def project(self, x_c: NDArray[(4, Any), float], **kwargs) \
            -> Tuple[NDArray[(3, Any), float], NDArray[(Any,), bool]]:
        raise NotImplementedError

    def unproject(self, uv: NDArray[(3, Any), float], **kwargs) \
            -> Tuple[NDArray[(4, Any), float], NDArray[(Any,), bool]]:
        raise NotImplementedError