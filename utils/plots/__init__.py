"""Plotting API. These are idempotent functions: read-only so that they can't change the data.
    They are called from the main.py file and from the terminal.
"""

from utils.plots.run import save_run_figures
from utils.plots.dataset import save_dataset_figures
from utils.plots.aggregate import save_batch_figures, categorize_run

__all__ = [
    "save_run_figures",
    "save_dataset_figures",
    "save_batch_figures",
    "categorize_run",
]
