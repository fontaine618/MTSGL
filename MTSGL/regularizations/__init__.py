"""regularizations module.

This module contains the classes defining various regularization applicable to Multi-task problems.
"""

__all__ = ["Regularization", "SparseGroupLasso"]
__version__ = '0.1'
__author__ = ['Simon Fontaine', 'Jinming Li', 'Yang Li']

from .regularization import *
from .sparsegrouplasso import *
