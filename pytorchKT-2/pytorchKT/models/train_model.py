import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
import pandas as pd
from pytorchKT.utils.utils import debug_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name

    if model_name in ["dkt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)

        loss = binary_cross_entropy(y.double(), t.double())
    elif model_name in ["sakt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())

    return loss


def model_forward(model: nn.Module, data, rel=None):
    model_name = model.model_name
    dcur = data
    q, c, r, t = (
        dcur["qseqs"].to(device),
        dcur["cseqs"].to(device),
        dcur["rseqs"].to(device),
        dcur["tseqs"].to(device),
    )
    qshft, cshft, rshft, tshft = (
        dcur["shft_qseqs"].to(device),
        dcur["shft_cseqs"].to(device),
        dcur["shft_rseqs"].to(device),
        dcur["shft_tseqs"].to(device),
    )
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)
    ys, preloss = [], []
    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)

    if model_name in ["dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)

    loss = cal_loss(model, ys, r, rshft, sm, preloss)
    return loss


def train_model(
    model,
    train_loader,
    valid_loader,
    num_epochs,
    opt,
    ckpt_path,
    test_loader=None,
    test_window_loader=None,
    save_model=False,
    data_config=None,
    fold=None,
):
    max_auc, best_epoch = 0, -1
    train_step = 0

    rel = None

    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step += 1
            model.train()
            loss = model_forward(model, data)
            opt.zero_grad()
            loss.backward()  # compute gradients
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip)
            opt.step()  # update model’s parameters

            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step % 10 == 0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text=text, fuc_name="train_model")
        loss_mean = np.mean(loss_mean)

        auc, acc = evaluate(model, valid_loader, model.model_name)

        if auc > max_auc + 1e-3:
            if save_model:
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_path, model.emb_type + "_model.ckpt"),
                )
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(
                        ckpt_path, model.emb_type + "_test_predictions.txt"
                    )
                    testauc, testacc = evaluate(
                        model, test_loader, model.model_name, save_test_path
                    )
                if test_window_loader != None:
                    save_test_path = os.path.join(
                        ckpt_path, model.emb_type + "_test_window_predictions.txt"
                    )
                    window_testauc, window_testacc = evaluate(
                        model, test_window_loader, model.model_name, save_test_path
                    )
            validauc, validacc = auc, acc
        print(
            f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}"
        )

        if test_loader != None:
            print(
                f"\ntestauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}"
            )

        if i - best_epoch >= 10:
            break
    return (
        testauc,
        testacc,
        window_testauc,
        window_testacc,
        validauc,
        validacc,
        best_epoch,
    )
