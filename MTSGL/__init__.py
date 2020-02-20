"""MTSGL package.

This package solves regularized Multi-task problems.
"""

__all__ = ["data", "proximal", "losses", "solvers", "regularizations", "fit"]
__version__ = '0.1'
__author__ = ['Simon Fontaine', 'Jinming Li', 'Yang Li']

import MTSGL.data
import MTSGL.proximal
import MTSGL.losses
import MTSGL.solvers
import MTSGL.regularizations
import MTSGL.fit