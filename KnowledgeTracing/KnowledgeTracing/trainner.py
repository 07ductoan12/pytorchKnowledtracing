import os
import argparse
import json
import pickle
import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from KnowledgeTracing.data_loaders import ASSIST2009
from KnowledgeTracing.models import DKT, DKTPlus, SAKT, PAM, MHA
from KnowledgeTracing.utils import collate_fn


def train(model_name, dataset_name):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json", encoding="utf-8") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    seq_len = train_config["seq_len"]

    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    # elif dataset_name == "ASSIST2015":
    #     dataset = ASSIST2015(seq_len)
    # elif dataset_name == "Algebra2005":
    #     dataset = Algebra2005(seq_len)
    # elif dataset_name == "Statics2011":
    #     dataset = Statics2011(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(os.path.join(ckpt_path, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=4)

    if model_name == "dkt":
        model = DKT(dataset.num_q, **model_config).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(dataset.num_q, **model_config).to(device)
    # elif model_name == "dkvmn":
    #     model = DKVMN(dataset.num_q, **model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(dataset.num_q, **model_config).to(device)
    elif model_name == "gkt":
        if model_config["method"] == "PAM":
            model = PAM(dataset.num_q, **model_config).to(device)
        elif model_config["method"] == "MHA":
            model = MHA(dataset.num_q, **model_config).to(device)
    else:
        print("The wrong model name was used...")
        return

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
        with open(os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb") as f:
            train_dataset.indices = pickle.load(f)
        with open(os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb") as f:
            test_dataset.indices = pickle.load(f)
    else:
        with open(os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb") as f:
            pickle.dump(train_dataset.indices, f)
        with open(os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb") as f:
            pickle.dump(test_dataset.indices, f)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True, collate_fn=collate_fn
    )

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    aucs, loss_means = model.train_model(
        train_loader, test_loader, num_epochs, opt, ckpt_path, device
    )

    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)
