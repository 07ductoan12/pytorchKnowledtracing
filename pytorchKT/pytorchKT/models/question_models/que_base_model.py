import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn import metrics


emb_type_list = ["qc_merge", "qid", "qaid", "qcid_merge"]
emb_type_map = {
    "akt-iekt": "qc_merge",
    "iekt-qid": "qc_merge",
    "iekt-qc_merge": "qc_merge",
    "iekt_ce-qid": "qc_merge",
    "dkt_que-qid": "qaid_qc",
    "dkt_que-qcaid": "qcaid",
    "dkt_que-qcaid_h": "qcaid_h",
}


class QueEmb(nn.Module):
    def __init__(
        self,
        num_q,
        num_c,
        emb_size,
        model_name,
        device="cpu",
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
    ) -> None:
        super().__init__()
        self.device = device
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        tmp_emb_type = f"{model_name}-{emb_type}"
        emb_type = emb_type_map.get(
            tmp_emb_type, tmp_emb_type.replace(f"{model_name}-", "")
        )
        print(f"emb_type is {emb_type}")
        self.emb_type = emb_type
        self.emb_path = emb_path
        self.pretrain_dim = pretrain_dim

        if emb_type in ["qc_merge", "qaid_qc"]:
            self.concept_emb = nn.Parameter(
                torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True
            )
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)
            self.que_c_linear = nn.Linear(self.emb_size * 2, self.emb_size)

        if emb_type == "qaid_c":
            self.que_c_linear = nn.Linear(2 * self.emb_size, self.emb_size)

        if emb_type in ["qcaid", "qcaid_h"]:
            self.concept_emb = nn.Parameter(
                torch.randn(self.num_c * 2, self.emb_size).to(device),
                requires_grad=True,
            )  # concept embeding
            self.que_inter_emb = nn.Embedding(self.num_q * 2, self.emb_size)
            self.que_c_linear = nn.Linear(2 * self.emb_size, self.emb_size)

        if emb_type.startswith("qaid"):
            self.interaction_emb = nn.Embedding(self.num_q * 2, self.emb_size)

        if emb_type.startswith("qid"):
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)

        if emb_type == "qcid":  # question_emb concat avg(concepts emb)
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)
            self.concept_emb = nn.Parameter(
                torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True
            )  # concept embeding
            self.que_c_linear = nn.Linear(2 * self.emb_size, self.emb_size)

        if emb_type == "iekt":
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)  # question embeding
            # self.que_emb.weight.requires_grad = False
            self.concept_emb = nn.Parameter(
                torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True
            )  # concept embeding
            self.que_c_linear = nn.Linear(2 * self.emb_size, self.emb_size)

        self.output_emb_dim = emb_size

    def get_avg_skill_emb(self, c):
        # add zero for padding
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.emb_size).to(self.device), self.concept_emb], dim=0
        )
        # shift c

        related_concepts = (c + 1).long()
        # [batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(axis=-2)

        # [batch_size, seq_len,1]
        concept_num = (
            torch.where(related_concepts != 0, 1, 0).sum(axis=-1).unsqueeze(-1)
        )
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = concept_emb_sum / concept_num
        return concept_avg

    def forward(self, q, c, r=None):
        emb_type = self.emb_type
        if "qc_merge" in emb_type:
            concept_avg = self.get_avg_skill_emb(c)
            que_emb = self.que_emb(q)
            que_c_emb = torch.cat([concept_avg, que_emb], dim=-1)

        if emb_type == "qc_merge":
            xemb = que_c_emb

        return xemb


class QueBaseModel(nn.Module):
    def __init__(self, model_name, emb_type, emb_path, pretrain_dim, device):
        super().__init__()
        self.model_name = model_name
        self.emb_type = emb_type
        self.emb_path = emb_path
        self.pretrain_dim = pretrain_dim
        self.device = device

    def compile(self, optimizer, lr=0.001, loss="binary_crossentropy", metric=None):
        self.lr = lr
        self.opt = self._get_optimizer(optimizer)
        self.loss_func = self._get_loss_func(loss)

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
        else:
            loss_func = loss
        return loss_func

    def _get_optimizer(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "gd":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            elif optimizer == "adagrad":
                optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
            elif optimizer == "adadelta":
                optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr)
            elif optimizer == "adam":
                optimizer = torch.optim.Adam(self.model.parameter(), lr=self.lr)
            else:
                raise ValueError("Unknown Optimizer: " + optimizer)
        return optimizer

    def train_one_step(self, data, process=True):
        raise NotImplementedError()

    def predict_one_step(self, data, process=True):
        raise NotImplementedError()

    def get_loss(self, ys, rshft, sm):
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft, sm)
        loss = self.loss_func(y_pred.double(), y_true.double())
        return loss

    def _save_model(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, self.model.emb_type + "_model.ckpt"),
        )

    def load_model(self, save_dir):
        net = torch.load(os.path.join(save_dir, self.emb_type + "_model.ckpt"))
        self.model.load_state_dict(net)

    def batch_to_device(self, data, process=True):
        if not process:
            return data
        dcur = data
        data_new = {}
        data_new["cq"] = torch.cat((dcur["qseqs"][:, 0:1], dcur["shft_qseqs"]), dim=1)
        data_new["cc"] = torch.cat((dcur["cseqs"][:, 0:1], dcur["shft_cseqs"]), dim=1)
        data_new["cr"] = torch.cat((dcur["rseqs"][:, 0:1], dcur["shft_rseqs"]), dim=1)
        data_new["ct"] = torch.cat((dcur["tseqs"][:, 0:1], dcur["shft_tseqs"]), dim=1)
        data_new["q"] = dcur["qseqs"]
        data_new["c"] = dcur["cseqs"]
        data_new["r"] = dcur["rseqs"]
        data_new["t"] = dcur["tseqs"]
        data_new["qshft"] = dcur["shft_qseqs"]
        data_new["cshft"] = dcur["shft_cseqs"]
        data_new["rshft"] = dcur["shft_rseqs"]
        data_new["tshft"] = dcur["shft_tseqs"]
        data_new["m"] = dcur["masks"]
        data_new["sm"] = dcur["smasks"]
        return data_new

    def train(
        self,
        train_dataset,
        valid_dataset,
        batch_size=16,
        valid_batch_size=None,
        num_epochs=32,
        test_loader=None,
        test_window_loader=None,
        save_dir="tmp",
        save_model=False,
        patient=10,
        shuffle=True,
        process=True,
    ):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        if valid_batch_size is None:
            valid_batch_size = batch_size

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )

        max_auc, best_epoch = 0, -1
        train_step = 0

        for i in range(1, num_epochs + 1, 1):
            loss_mean = []
            pass
