import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


device = "cpu" if not torch.cuda.is_available() else "cuda"


class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
            # Dropout(self.dropout),
        )

    def forward(self, in_fea):
        return self.FFN(in_fea)


def ut_mask(seq_len):
    """Upper Triangular Mask"""
    return (
        torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        .to(dtype=torch.bool)
        .to(device)
    )


def lt_mask(seq_len):
    """Upper Triangular Mask"""
    return (
        torch.tril(torch.ones(seq_len, seq_len), diagonal=-1)
        .to(dtype=torch.bool)
        .to(device)
    )


def pos_encode(seq_len):
    """position Encoding"""
    return torch.arange(seq_len).unsqueeze(0).to(device)


def get_clones(module, N):
    """Cloning nn modules"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def save_cur_predict_result(dres, q, r, d, t, m, sm, p):
    """_summary_

    Args:
        dres (_type_): _description_
        q (_type_): _description_
        r (_type_): _description_
        d (_type_): _description_
        t (_type_): _description_
        m (_type_): _description_
        sm (_type_): _description_
        p (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []

    for i in range(0, t.shape[0]):
        cps = torch.masked_select(p[i], sm[i]).detach().cpu()
        cts = torch.masked_select(t[i], sm[i]).detach().cpu()

        cqs = torch.masked_select(q[i], m[i]).detach().cpu()
        crs = torch.masked_select(r[i], m[i]).detach().cpu()

        cds = torch.masked_select(d[i], sm[i]).detach().cpu()

        qs, rs, ts, ps, ds = [], [], [], [], []
        for cq, cr in zip(cqs.int(), crs.int()):
            qs.append(cq.item())
            rs.append(cr.item())
        for ct, cp, cd in zip(cts.int(), cps, cds.int()):
            ts.append(ct.item())
            ps.append(cp.item())
            ds.append(cd.item())

        try:
            auc = metrics.roc_auc_score(y_true=np.array(ts), y_score=np.array(ps))
        except Exception as e:
            auc = -1
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        dres[len(dres)] = [qs, rs, ds, ts, ps, prelabels, auc, acc]
        results.append(str([qs, rs, ds, ts, ps, prelabels, auc, acc]))
    return "\n".join(results)


def save_question_res(dres, fout, early=False):
    # print(f"dres: {dres.keys()}")
    # qidxs, late_trues, late_mean, late_vote, late_all, early_trues, early_preds
    for i in range(0, len(dres["qidxs"])):
        row, qidx, qs, cs, lt, lm, lv, la = (
            dres["row"][i],
            dres["qidxs"][i],
            dres["questions"][i],
            dres["concepts"][i],
            dres["late_trues"][i],
            dres["late_mean"][i],
            dres["late_vote"][i],
            dres["late_all"][i],
        )
        conceptps = dres["concept_preds"][i]
        curres = [row, qidx, qs, cs, conceptps, lt, lm, lv, la]
        if early:
            et, ep = dres["early_trues"][i], dres["early_preds"][i]
            curres = curres + [et, ep]
        curstr = "\t".join(
            [
                str(round(s, 4))
                if type(s) == type(0.1) or type(s) == np.float32
                else str(s)
                for s in curres
            ]
        )
        fout.write(curstr + "\n")


def save_each_question_res(dcres, dqres, ctrues, cpreds):
    high, low = [], []
    for true, pred in zip(ctrues, cpreds):
        dcres["trues"].append(true)
        dcres["preds"].append(pred)
        if pred >= 0.5:
            high.append(pred)
        else:
            low.append(pred)
    cpreds = np.array(cpreds)

    late_mean = np.mean(cpreds)
    correctnum = list(cpreds >= 0.5).count(True)
    late_vote = np.mean(high) if correctnum / len(cpreds) >= 0.5 else np.mean(low)
    late_all = np.mean(high) if correctnum == len(cpreds) else np.mean(low)

    assert len(set(ctrues)) == 1

    dqres["trues"].append(dcres["trues"][-1])
    dqres["late_mean"].append(late_mean)
    dqres["late_vote"].append(late_vote)
    dqres["late_all"].append(late_all)
    return late_mean, late_vote, late_all


def cal_predres(dcres, dqres):
    """_summary_

    Args:
        dcres (_type_): _description_
        dqres (_type_): _description_

    Returns:
        _type_: _description_
    """
    dres = dict()  # {"concept": [], "late_mean": [], "late_vote": [], "late_all": []}

    ctrues, cpreds = np.array(dcres["trues"]), np.array(dcres["preds"])
    auc = metrics.roc_auc_score(y_true=ctrues, y_score=cpreds)
    predlabels = [1 if p >= 0.5 else 0 for p in cpreds]
    acc = metrics.accuracy_score(ctrues, predlabels)

    dres["concepts"] = [len(cpreds), auc, acc]

    qtrues = np.array(dqres["trues"])
    for key in dqres:
        if key == "trues":
            continue

        preds = np.array(dqres[key])
        auc = metrics.roc_auc_score(y_true=qtrues, y_score=preds)
        predlabels = [1 if p >= 0.5 else 0 for p in preds]
        acc = metrics.accuracy_score(qtrues, predlabels)
        dres[key] = [len(preds), auc, acc]
    return dres


def save_currow_question_res(idx, dcres, dqres, qidxs, ctrues, cpreds, uid, fout):
    """_summary_

    Args:
        idx (_type_): _description_
        dcres (_type_): _description_
        dqres (_type_): _description_
        qidxs (_type_): _description_
        ctrues (_type_): _description_
        cpreds (_type_): _description_
        uid (_type_): _description_
        fout (_type_): _description_
    """

    dqidx = dict()

    for i, qidx in enumerate(qidxs):
        true, pred = ctrues[i], cpreds[i]
        dqidx.setdefault(qidx, {"trues": [], "preds": []})
        dqidx[qidx]["trues"].append(true)
        dqidx[qidx]["preds"].append(pred)

    for qidx, values in dqidx.items():
        ctrues, cpreds = values["trues"], values["preds"]
        late_mean, late_vote, late_all = save_each_question_res(
            dcres, dqres, ctrues, cpreds
        )
        fout.write(
            "\t".join(
                [
                    str(idx),
                    str(uid),
                    str(qidx),
                    str("|"),
                    str(late_mean),
                    str(late_vote),
                    str(late_all),
                ]
            )
            + "\n"
        )
    fout.write("\n")
