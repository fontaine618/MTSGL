"""data module.

This module contains the classes defining data structures and utilities to translate data frames to them.
"""

__all__ = ["Loss", "WLS", "MTLoss"]
__version__ = '0.1'
__author__ = ['Simon Fontaine', 'Jinming Li', 'Yang Li']

from .Loss import *
from .WLS import *
from .MTLoss import *


