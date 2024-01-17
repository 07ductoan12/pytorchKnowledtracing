import os, sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime


def get_now_time():
    """Return the time string, the format is %Y-%m-%d %H:%M:%S

    Returns:
        str: now time
    """
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return dt_string


def debug_print(text, fuc_name=""):
    """Printing text with function name.

    Args:
        text (str): the text will print
        fuc_name (str, optional): _description_. Defaults to "".
    """
    print(f"{get_now_time()} - {fuc_name} - said: {text}")
