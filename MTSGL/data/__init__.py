"""data module.

This module contains the classes defining data structures and utilities to translate data frames to them.
"""

__all__ = ["utils", "Data", "MultivariateData", "MultiTaskData"]
__version__ = '0.1'
__author__ = ['Simon Fontaine', 'Jinming Li', 'Yang Li']

from .utils import *
from .data import *
from .multivariatedata import *
from .multitaskdata import *


