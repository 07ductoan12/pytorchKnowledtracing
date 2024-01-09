import torch
import numpy as np
import os
from .dkt import DKT
from .sakt import SAKT

device = "cpu" if not torch.cuda.is_available() else "cuda"


def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "dkt":
        model = DKT(
            data_config["num_c"],
            emb_size=model_config["emb_size"],
            dropout=model_config["dropout"],
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
        ).to(device)
    elif model_name == "sakt":
        model = SAKT(
            data_config["num_c"],
            seq_len=model_config["seq_len"],
            emb_size=model_config["emb_size"],
            num_attn_heads=model_config["num_attn_heads"],
            dropout=model_config["dropout"],
            num_en=model_config["num_en"],
            emb_type=emb_type,
        ).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model


def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type + "_model.ckpt"))
    model.load_state_dict(net)
    return model
