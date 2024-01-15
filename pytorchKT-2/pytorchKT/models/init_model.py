import torch
import os
from pytorchKT.models.kc_models import DKT, SAKT, DKVMN, DIMKT, DKTPlus

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"


def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "dkt":
        model = DKT(
            data_config["num_c"],
            emb_size=model_config["emb_size"],
            dropout=model_config["dropout"],
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
        ).to(DEVICE)
    elif model_name == "dkt+":
        model = DKTPlus(
            data_config["num_c"],
            emb_size=model_config["emb_size"],
            lambda_r=model_config["lambda_r"],
            lambda_w1=model_config["lambda_w1"],
            lambda_w2=model_config["lambda_w2"],
            dropout=model_config["dropout"],
            emb_type=emb_type,
        ).to(DEVICE)
    elif model_name == "sakt":
        model = SAKT(
            data_config["num_c"],
            seq_len=model_config["seq_len"],
            emb_size=model_config["emb_size"],
            num_attn_heads=model_config["num_attn_heads"],
            dropout=model_config["dropout"],
            num_en=model_config["num_en"],
            emb_type=emb_type,
        ).to(DEVICE)
    elif model_name == "dkvmn":
        model = DKVMN(
            data_config["num_c"],
            dim_s=model_config["dim_s"],
            size_m=model_config["size_m"],
            emb_type=emb_type,
        ).to(DEVICE)
    elif model_name == "dimkt":
        model = DIMKT(
            data_config["num_q"],
            data_config["num_c"],
            dropout=model_config["dropout"],
            emb_size=model_config["emb_size"],
            batch_size=model_config["batch_size"],
            num_steps=model_config["num_steps"],
            difficult_levels=model_config["difficult_levels"],
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
        ).to(DEVICE)
    else:
        print("The wrong model name was used...")
        return None
    return model


def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type + "_model.ckpt"))
    model.load_state_dict(net)
    return model
