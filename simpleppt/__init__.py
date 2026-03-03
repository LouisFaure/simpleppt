from importlib.metadata import version as _version

__version__ = _version(__name__)
del _version

from .load_data import load_example
from .ppt import ppt
from .project_ppt import project_ppt
from .SimplePPT import SimplePPT
