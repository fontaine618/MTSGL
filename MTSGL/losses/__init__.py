"""losses module.

This module contains the classes defining losses.
"""

__all__ = ["Loss", "WLS", "MTLoss", "SeparableMTLoss", "MTWLS"]
__version__ = '0.1'
__author__ = ['Simon Fontaine', 'Jinming Li', 'Yang Li']

from .loss import Loss
from .wls import WLS
from .mtloss import MTLoss, SeparableMTLoss, MTWLS


