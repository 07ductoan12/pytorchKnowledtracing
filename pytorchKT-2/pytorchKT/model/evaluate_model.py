import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from sklearn import metrics
import pandas as pd
import csv

device = "cpu" if not torch.cuda.is_available() else "cuda"


def save_cur_predict_result(dres, q, r, d, t, m, sm, p):
    # dres, q, r, qshft, rshft, m, sm, y
    results = []
    for i in range(0, t.shape[0])
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
            # print(e)
            auc = -1
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        dres[len(dres)] = [qs, rs, ds, ts, ps, prelabels, auc, acc]
        results.append(str([qs, rs, ds, ts, ps, prelabels, auc, acc]))
    return "\n".join(results)

def evaluate(model, test_loader, model_name, rel=None, save_path=""):
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")

    with torch.no_grad():
        y_trues = []
        y_scores = []
        dres = dict()
        test_mini_index = 0

        for data in test_loader:
            if model_name in  ["dkt_forget", "bakt_time"]:
                dcur, dgaps = data
            else:
                dcur = data
            q, c, r = dcur['qseqs'], dcur['cseqs'], dcur['rseqs']
            qshft, cshft, rshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"]

            m, sm = dcur["masks"], dcur["smasks"]
            q, c, r, qshft, cshft, rshft, m, sm = q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device)
            model.eval()
    
