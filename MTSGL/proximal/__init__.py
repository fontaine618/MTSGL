"""proximal module.

This modules contains utility functions to compute various proximal operators."""

__all__ = ["proximal_sgl"]
__version__ = '0.1'
__author__ = ['Simon Fontaine', 'Jinming Li', 'Yang Li']

from .proximal_sgl import *
from ._projections import *

