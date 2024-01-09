import os, sys
import argparse
import json
import torch
import copy

from pytorchKT.models import evaluate_splitpred_question, load_model
from pytorchKT.arguments import get_eval_argments


def eval():
    args = get_eval_argments()

    print(args)
    params = vars(args)

    save_dir, use_pred, ratio = (
        params["save_dir"],
        params["use_pred"],
        params["train_ratio"],
    )

    with open(os.path.join(save_dir, "config.json"), encoding="utf-8") as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])

        for remove_item in ["use_wandb", "learning_rate", "add_uuid", "l2"]:
            if remove_item in model_config:
                del model_config[remove_item]

        trained_params = config["params"]
        model_name, dataset_name, emb_type = (
            trained_params["model_name"],
            trained_params["dataset_name"],
            trained_params["emb_type"],
        )
        seq_len = config["train_config"]["seq_len"]
        if model_name in ["saint", "sakt", "atdkt"]:
            model_config["seq_len"] = seq_len
        data_config = config["data_config"]

    print(
        f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}"
    )
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")
    use_pred = True if use_pred == 1 else False

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    print(f"Start predict use_pred: {use_pred}, ratio: {ratio}...")
    atkt_pad = True if params["atkt_pad"] == 1 else False
    if model_name == "atkt":
        save_test_path = os.path.join(
            save_dir,
            model.emb_type
            + "_test_ratio"
            + str(ratio)
            + "_"
            + str(use_pred)
            + "_"
            + str(atkt_pad)
            + "_predictions.txt",
        )
    else:
        save_test_path = os.path.join(
            save_dir,
            model.emb_type
            + "_test_ratio"
            + str(ratio)
            + "_"
            + str(use_pred)
            + "_predictions.txt",
        )
    testf = os.path.join(data_config["dpath"], params["test_filename"])
    dfinal = evaluate_splitpred_question(
        model, data_config, testf, model_name, save_test_path, use_pred, ratio, atkt_pad
    )
    for key in dfinal:
        print(key, dfinal[key])
    dfinal.update(config["params"])


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        eval()
