import os, sys
import argparse
import json

import torch

torch.set_num_threads(4)
from torch.optim import SGD, Adam
import copy
from pytorchKT.models import train_model, init_model
from pytorchKT.datasets.create_dataloader import init_dataset4train
from pytorchKT.utils import debug_print
import datetime
from pytorchKT.arguments import get_train_arguments

device = "cpu" if not torch.cuda.is_available() else "cuda"


def save_config(train_config, model_config, data_config, params, save_dir):
    d = {
        "train_config": train_config,
        "model_config": model_config,
        "data_config": data_config,
        "params": params,
    }
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w", encoding="utf-8") as fout:
        json.dump(d, fout)


def train():
    params = vars(get_train_arguments())

    model_name, dataset_name, fold, emb_type, save_dir = (
        params["model_name"],
        params["dataset_name"],
        params["fold"],
        params["emb_type"],
        params["save_dir"],
    )

    debug_print(text="load config files.", fuc_name="main")

    with open("./pytorchKT/configs/kt_config.json", encoding="utf-8") as f:
        config = json.load(f)
        train_config = config["train_config"]
        keys = config[model_name].keys()
        if model_name in ["sakt"]:
            train_config["batch_size"] = 64  ## because of OOM
        model_config = copy.deepcopy(params)

        for key in keys:
            if not key in model_config.keys():
                model_config[key] = config[model_name][key]

        for key in [
            "model_name",
            "dataset_name",
            "emb_type",
            "save_dir",
            "fold",
            "seed",
        ]:
            del model_config[key]
        if "batch_size" in params:
            train_config["batch_size"] = params["batch_size"]
        if "num_epochs" in params:
            train_config["num_epochs"] = params["num_epochs"]

    batch_size, num_epochs, optimizer = (
        train_config["batch_size"],
        train_config["num_epochs"],
        train_config["optimizer"],
    )

    with open("./pytorchKT/configs/data_config.json") as fin:
        data_config = json.load(fin)

    if "maxlen" in data_config[dataset_name]:  # prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]["maxlen"]
    seq_len = train_config["seq_len"]

    print("Start init data")
    print(dataset_name, model_name, data_config[dataset_name], fold, batch_size)

    debug_print(text="init_dataset", fuc_name="main")
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(
            dataset_name, model_name, data_config, fold, batch_size
        )
    else:
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(
            dataset_name,
            model_name,
            data_config,
            fold,
            batch_size,
            diff_level=diff_level,
        )

    params_str = "_".join(
        [str(v) for k, v in params.items() if not k in ["other_config"]]
    )

    print(f"params: {params}, params_str: {params_str}")

    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(
        f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}"
    )
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    learning_rate = params["learning_rate"]

    if model_name in ["sakt"]:
        model_config["seq_len"] = seq_len

    debug_print(text="init_model", fuc_name="main")
    print(f"model_name:{model_name}")
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f"model is {model}")
    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    debug_print(text="train model", fuc_name="main")

    (
        testauc,
        testacc,
        window_testauc,
        window_testacc,
        validauc,
        validacc,
        best_epoch,
    ) = train_model(
        model,
        train_loader,
        valid_loader,
        num_epochs,
        opt,
        ckpt_path,
        None,
        None,
        save_model,
    )

    if save_model:
        best_model = init_model(
            model_name, model_config, data_config[dataset_name], emb_type
        )
        net = torch.load(os.path.join(ckpt_path, emb_type + "_model.ckpt"))
        best_model.load_state_dict(net)
    print(
        "fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch"
    )
    print(
        str(fold)
        + "\t"
        + model_name
        + "\t"
        + emb_type
        + "\t"
        + str(round(testauc, 4))
        + "\t"
        + str(round(testacc, 4))
        + "\t"
        + str(round(window_testauc, 4))
        + "\t"
        + str(round(window_testacc, 4))
        + "\t"
        + str(validauc)
        + "\t"
        + str(validacc)
        + "\t"
        + str(best_epoch)
    )
    save_config(
        train_config,
        model_config,
        data_config[dataset_name],
        params,
        save_dir=ckpt_path,
    )
    print(f"end:{datetime.datetime.now()}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        print("Start TRAIN process...\n")
        train()
