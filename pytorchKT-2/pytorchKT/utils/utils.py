import os, sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from pytorchKT.datasets.data_loader import KTDataset


def debug_print(text, fuc_name=""):
    """Printing text with function name.

    Args:
        text (str): the text will print
        fuc_name (str, optional): _description_. Defaults to "".
    """
    print(f"{get_now_time()} - {fuc_name} - said: {text}")
