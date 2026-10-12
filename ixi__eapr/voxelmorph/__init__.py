# ---- voxelmorph ----
# unsupervised learning for image registration

import sys
from pathlib import Path

# The original distribution vendors pystrum here. Make direct CLI entry points
# work without relying on an unrelated package being installed globally.
_external = Path(__file__).resolve().parent / "torch" / "external"
if str(_external) not in sys.path:
    sys.path.insert(0, str(_external))

from . import generators
from . import py
from .py.utils import default_unet_features


# import backend-dependent submodules
backend = py.utils.get_backend()
if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    from . import torch
    from .torch import layers
    from .torch import losses
    from .torch import iirp

