"""Drilling kick detection research prototype.

Modules in this package implement a reproducible pipeline for auditing the
DataDRILL kick dataset, building experimental proxy labels, training and
evaluating models, and producing research outputs.
"""

from .data_loader import PROJECT_ROOT, load_config, load_raw_data

__all__ = ["PROJECT_ROOT", "load_config", "load_raw_data"]
