import os
import numpy as np
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import csv
from torch.nn.functional import one_hot
from sklearn import metrics
import pandas as pd
from .utils import (
    save_cur_predict_result,
    save_each_question_res,
    save_question_res,
    cal_predres,
    save_currow_question_res,
)

que_type_models = []

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"


def evaluate(model, test_loader, model_name, rel=None, save_path=""):
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")
    with torch.no_grad():
        y_trues = []
        y_scores = []
        dres = dict()
        test_mini_index = 0
        for data in test_loader:
            if model_name in ["dkt_forget", "bakt_time"]:
                dcur, dgaps = data
            else:
                dcur = data
            if model_name in ["dimkt", "dimkt_cc"]:
                q, c, r, sd, qd = (
                    dcur["qseqs"],
                    dcur["cseqs"],
                    dcur["rseqs"],
                    dcur["sdseqs"],
                    dcur["qdseqs"],
                )
                qshft, cshft, rshft, sdshft, qdshft = (
                    dcur["shft_qseqs"],
                    dcur["shft_cseqs"],
                    dcur["shft_rseqs"],
                    dcur["shft_sdseqs"],
                    dcur["shft_qdseqs"],
                )
                sd, qd, sdshft, qdshft = (
                    sd.to(DEVICE),
                    qd.to(DEVICE),
                    sdshft.to(DEVICE),
                    qdshft.to(DEVICE),
                )
            else:
                q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"]
                qshft, cshft, rshft = (
                    dcur["shft_qseqs"],
                    dcur["shft_cseqs"],
                    dcur["shft_rseqs"],
                )
            m, sm = dcur["masks"], dcur["smasks"]
            q, c, r, qshft, cshft, rshft, m, sm = (
                q.to(DEVICE),
                c.to(DEVICE),
                r.to(DEVICE),
                qshft.to(DEVICE),
                cshft.to(DEVICE),
                rshft.to(DEVICE),
                m.to(DEVICE),
                sm.to(DEVICE),
            )
            model.eval()

            # print(f"before y: {y.shape}")
            cq = torch.cat((q[:, 0:1], qshft), dim=1)
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)
            if model_name in ["atdkt"]:
                """
                y = model(dcur)
                import pickle
                with open(f"{test_mini_index}_result.pkl",'wb') as f:
                    data = {"y":y,"cshft":cshft,"num_c":model.num_c,"rshft":rshft,"qshft":qshft,"sm":sm}
                    pickle.dump(data,f)
                """
                y = model(dcur)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["rkt"]:
                y, attn = model(dcur, rel)
                y = y[:, 1:]
                if q.numel() > 0:
                    c, cshft = q, qshft  # question level
            elif model_name in ["bakt_time"]:
                y = model(dcur, dgaps)
                y = y[:, 1:]
            elif model_name in ["simplekt", "sparsekt"]:
                y = model(dcur)
                y = y[:, 1:]
            elif model_name in ["dkt", "dkt+"]:
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkt_forget"]:
                y = model(c.long(), r.long(), dgaps)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkvmn", "deep_irt", "skvmn", "deep_irt"]:
                y = model(cc.long(), cr.long())
                y = y[:, 1:]
            elif model_name in ["kqn", "sakt"]:
                y = model(c.long(), r.long(), cshft.long())
            elif model_name == "saint":
                y = model(cq.long(), cc.long(), r.long())
                y = y[:, 1:]
            elif model_name in ["atkt", "atktfix"]:
                y, _ = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name == "hawkes":
                ct = torch.cat((dcur["tseqs"][:, 0:1], dcur["shft_tseqs"]), dim=1)
                y = model(cc.long(), cq.long(), ct.long(), cr.long())  # , csm.long())
                y = y[:, 1:]
            elif model_name in que_type_models and model_name != "lpkt":
                y = model.predict_one_step(data)
                c, cshft = q, qshft  # question level
            elif model_name == "dimkt":
                y = model(
                    q.long(),
                    c.long(),
                    sd.long(),
                    qd.long(),
                    r.long(),
                    qshft.long(),
                    cshft.long(),
                    sdshft.long(),
                    qdshft.long(),
                )
            elif model_name in ["dimkt_cc"]:
                y = model(
                    c.long(),
                    sd.long(),
                    r.long(),
                    cshft.long(),
                    sdshft.long(),
                )
            # save predict result
            if save_path != "":
                result = save_cur_predict_result(dres, c, r, cshft, rshft, m, sm, y)
                fout.write(result + "\n")

            y = torch.masked_select(y, sm).detach().cpu()
            t = torch.masked_select(rshft, sm).detach().cpu()

            y_trues.append(t.numpy())
            y_scores.append(y.numpy())
            test_mini_index += 1
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)

        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)

    return auc, acc


def early_fusion(curhs, model, model_name):
    if model_name in ["sakt"]:
        p = torch.sigmoid(model.pred(model.dropout_layer(curhs[0]))).squeeze(-1)
    return p


def late_fusion(dcur: dict, curdf, fusion_type=["mean", "vote", "all"]):
    high, low = [], []
    for pred in curdf["preds"]:
        if pred >= 0.5:
            high.append(pred)
        else:
            low.append(pred)

    if "mean" in fusion_type:
        dcur.setdefault("late_mean", [])
        dcur["late_mean"].append(round(float(curdf["preds"].mean()), 4))
    if "vote" in fusion_type:
        dcur.setdefault("late_vote", [])
        correctnum = list(curdf["preds"] >= 0.5).count(True)
        late_vote = (
            np.mean(high) if correctnum / len(curdf["preds"]) >= 0.5 else np.mean(low)
        )
        dcur["late_vote"].append(late_vote)

    if "all" in fusion_type:
        dcur.setdefault("late_vote", [])
        late_all = np.mean(high) if correctnum == len(curdf["preds"]) else np.mean(low)
        dcur["late_all"].append(late_all)
        return


def effective_fusion(df: pd.DataFrame, model, model_name, fusion_type):
    dres = dict()
    df = df.groupby("qids", as_index=True, sort=True)

    curhs, curr = [[], []], []
    dcur = {
        "late_trues": [],
        "qidxs": [],
        "questions": [],
        "concepts": [],
        "row": [],
        "concept_preds": [],
    }

    hasearly = ["sakt"]
    for ui in df:
        curdf = ui[1]
        if model_name in hasearly and model_name not in ["kqn", "lpkt", "deep_irt"]:
            curhs[0].append(curdf["hidden"].mean().astype(float))
        else:
            pass

        curr.append(int(curdf["response"].mean()))
        dcur["late_trues"].append(int(curdf["response"].mean()))
        dcur["qidxs"].append(ui[0])
        dcur["row"].append(int(curdf["row"].mean()))
        dcur["questions"].append(
            ",".join([str(int(s)) for s in curdf["questions"].tolist()])
        )
        dcur["concepts"].append(
            ",".join([str(int(s)) for s in curdf["concepts"].tolist()])
        )
        late_fusion(dcur, curdf)

    for key, values in dcur.items():
        dres.setdefault(key, [])
        dres[key].append(np.array(values))

    if "early_fusion" in fusion_type and model_name in hasearly:
        curhs = [torch.tensor(curh).float().cpu() for curh in curhs]
        curr = torch.tensor(curr).long().to(DEVICE)

        p = early_fusion(curhs, model, model_name)
        dres.setdefault("early_trues", [])
        dres["early_trues"].append(curr.cpu().numpy())
        dres.setdefault("early_preds", [])
        dres["early_preds"].append(p.cpu().numpy())
    return dres


def group_fusion(dmerge, model, model_name, fusion_type, fout):
    hs, sms, cq, cc, rs, ps, qidxs, rests, orirows = (
        dmerge["hs"],
        dmerge["sm"],
        dmerge["cq"],
        dmerge["cc"],
        dmerge["cr"],
        dmerge["y"],
        dmerge["qidxs"],
        dmerge["rests"],
        dmerge["orirow"],
    )
    if cq.shape[1] == 0:
        cq = cc

    hasearly = ["sakt"]
    alldfs, drest = [], dict()

    for bz in range(rs.shape[0]):
        cursm = [0] + sms[bz].cpu().tolist()
        curqidxs = [-1] + qidxs[bz].cpu().tolist()
        currests = [-1] + rests[bz].cpu().tolist()
        currows = [-1] + orirows[bz].cpu().tolist()
        curps = [-1] + ps[bz].cpu().tolist()
        # print(f"qid: {len(curqidxs)}, select: {len(cursm)}, response: {len(rs[bz].cpu().tolist())}, preds: {len(curps)}")
        df = pd.DataFrame(
            {
                "qidx": curqidxs,
                "rest": currests,
                "row": currows,
                "select": cursm,
                "questions": cq[bz].cpu().tolist(),
                "concepts": cc[bz].cpu().tolist(),
                "response": rs[bz].cpu().tolist(),
                "preds": curps,
            }
        )

        if model_name in hasearly and model_name not in ["kqn", "lpkt", "deep_irt"]:
            df["hidden"] = [np.array(a) for a in hs[0][bz].cpu().tolist()]

        df = df[df["select"] != 0]
        alldfs.append(df)

    effective_dfs, rest_start = [], -1
    flag = False

    for i in range(len(alldfs) - 1, -1, -1):
        df = alldfs[i]
        counts = (df["rest"] == 0).value_counts()

        if not flag and False not in counts:
            flag = True
            effective_dfs.append(df)
            rest_start = i + 1
        elif flag:
            effective_dfs.append(df)
    if rest_start == -1:
        rest_start = 0

    for key in dmerge.keys():
        if key == "hs":
            drest[key] = []
            if model_name in hasearly and model_name not in ["kqn", "lpkt", "deep_irt"]:
                drest[key] = [dmerge[key][0][rest_start:]]
            elif model_name in ["kqn", "lpkt", "deep_irt"]:
                drest[key] = [dmerge[key][0][rest_start:], dmerge[key][1][rest_start:]]
        else:
            drest[key] = dmerge[key][rest_start:]

    dfs = dict()
    for df in effective_dfs:
        for i, row in df.iterrows():
            for key in row.keys():
                dfs.setdefault(key, [])
                dfs[key].extend([row[key]])
    df = pd.DataFrame(dfs)

    if df.shape[0] == 0:
        return {}, drest

    dres = effective_fusion(df, model, model_name, fusion_type)

    dfinal = dict()
    for key, value in dres.items():
        dfinal[key] = np.concatenate(value, axis=0)
    early = False
    if model_name in hasearly and "early_fusion" in fusion_type:
        early = True
    save_question_res(dfinal, fout, early)
    return dfinal, drest


def evaluate_question(
    model,
    test_loader,
    model_name,
    fusion_type=["early_fusion", "late_fusion"],
    save_path="",
):
    hasearly = ["sakt"]
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")
        if model_name in hasearly:
            fout.write(
                "\t".join(
                    [
                        "orirow",
                        "qidx",
                        "questions",
                        "concepts",
                        "concept_preds",
                        "late_trues",
                        "late_mean",
                        "late_vote",
                        "late_all",
                        "early_trues",
                        "early_preds",
                    ]
                )
                + "\n"
            )
        else:
            fout.write(
                "\t".join(
                    [
                        "orirow",
                        "qidx",
                        "questions",
                        "concepts",
                        "concept_preds",
                        "late_trues",
                        "late_mean",
                        "late_vote",
                        "late_all",
                    ]
                )
                + "\n"
            )

    with torch.no_grad():
        dinfos = dict()
        dhistory = dict()
        history_keys = ["hs", "sm", "cq", "cc", "cr", "y", "qidxs", "rests", "orirow"]
        y_trues, y_scores = [], []
        lenc = 0
        for data in test_loader:
            dcurori, dqtest = data

            q, c, r = dcurori["qseqs"], dcurori["cseqs"], dcurori["rseqs"]
            qshft, cshft, rshft = (
                dcurori["shft_qseqs"],
                dcurori["shft_cseqs"],
                dcurori["shft_rseqs"],
            )

            m, sm = dcurori["masks"], dcurori["smasks"]
            q, c, r, qshft, cshft, rshft, m, sm = (
                q.to(DEVICE),
                c.to(DEVICE),
                r.to(DEVICE),
                qshft.to(DEVICE),
                cshft.to(DEVICE),
                rshft.to(DEVICE),
                m.to(DEVICE),
                sm.to(DEVICE),
            )
            qidxs, rests, orirow = dqtest["qidxs"], dqtest["rests"], dqtest["orirow"]
            lenc += q.shape[0]
            model.eval()

            cq = torch.cat((q[:, 0:1], qshft), dim=1)
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)
            dcur = dict()

            if model_name == "sakt":
                y, h = model(c.long(), r.long(), cshft.long(), True)
            elif model_name in ["dkt", "dkt+"]:
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)

            concepty = torch.masked_select(y, sm).detach().cpu()
            conceptt = torch.masked_select(rshft, sm).detach().cpu()

            y_trues.append(conceptt.numpy())
            y_scores.append(concepty.numpy())

            hs = []

            if model_name in hasearly:
                hs = [h]

            (
                dcur["hs"],
                dcur["sm"],
                dcur["cq"],
                dcur["cc"],
                dcur["cr"],
                dcur["y"],
                dcur["qidxs"],
                dcur["rests"],
                dcur["orirow"],
            ) = (hs, sm, cq, cc, cr, y, qidxs, rests, orirow)
            # merge history
            dmerge = dict()

            for key in history_keys:
                if len(dhistory) == 0:
                    dmerge[key] = dcur[key]
                else:
                    dmerge[key] = torch.cat((dhistory[key], dcur[key]), dim=0)

            dcur, dhistory = group_fusion(dmerge, model, model_name, fusion_type, fout)
            for key, value in dcur.items():
                dinfos.setdefault(key, [])
                dinfos[key].append(value)

            if "early_fusion" in dinfos and "late_fusion" in dinfos:
                assert dinfos["early_trues"][-1].all() == dinfos["late_trues"][-1].all()

        aucs, accs = dict(), dict()
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        # print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        aucs["concepts"] = auc
        accs["concepts"] = acc

        for key, values in dinfos.items():
            if key not in ["late_mean", "late_vote", "late_all", "early_preds"]:
                continue
            ts = np.concatenate(
                values["late_trues"], axis=0
            )  # early_trues == late_trues
            ps = np.concatenate(dinfos[key], axis=0)
            # print(f"key: {key}, ts.shape: {ts.shape}, ps.shape: {ps.shape}")
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
            prelabels = [1 if p >= 0.5 else 0 for p in ps]
            acc = metrics.accuracy_score(ts, prelabels)
            aucs[key] = auc
            accs[key] = acc
    return aucs, accs


def get_cur_teststart(is_repeat: List[int], train_ratio: float) -> Tuple[int, int, int]:
    """_summary_

    Args:
        is_repeat (List[int]): list check question is repeate
        train_ratio (float): split ration

    Returns:
        Tuple[int, int, int]: _description_
    """
    curl = len(is_repeat)
    qlen = is_repeat.count(0)

    qtrainlen = int(qlen * train_ratio)
    qtrainlen = 1 if qtrainlen == 0 else qtrainlen
    qtrainlen = qtrainlen - 1 if qtrainlen == qlen else qtrainlen

    ctrainlen, qidx = 0, 0
    i = 0
    while i < curl:  # tìm idx kết thúc idx train
        if is_repeat[i] == 0:
            qidx += 1

        if qidx == qtrainlen:
            break
        i += 1

    for j in range(i + 1, curl):  # skip repeat
        if is_repeat[j] == 0:
            ctrainlen = j
            break
    return qlen, qtrainlen, ctrainlen


def log2(t):
    import math

    return round(math.log(t + 1, 2))


def calC(row, data_config):
    repeated_gap, sequence_gap, past_counts = [], [], []
    # uid = row["uid"]
    # default: concepts
    skills = row["concepts"].split(",")
    timestamps = row["timestamps"].split(",")
    dlastskill, dcount = dict(), dict()
    pret = None
    idx = -1
    for s, t in zip(skills, timestamps):
        idx += 1
        s, t = int(s), int(t)
        if s not in dlastskill or s == -1:
            cur_repeated_gap = 0
        else:
            cur_repeated_gap = log2((t - dlastskill[s]) / 1000 / 60) + 1  # minutes
        dlastskill[s] = t

        repeated_gap.append(cur_repeated_gap)
        if pret is None or t == -1:
            cur_last_gap = 0
        else:
            cur_last_gap = log2((t - pret) / 1000 / 60) + 1
        pret = t
        sequence_gap.append(cur_last_gap)

        dcount.setdefault(s, 0)
        ccount = log2(dcount[s])
        ccount = (
            data_config["num_pcount"] - 1
            if ccount >= data_config["num_pcount"]
            else ccount
        )
        past_counts.append(ccount)

        dcount[s] += 1
    return repeated_gap, sequence_gap, past_counts


def get_info_dkt_forget(row, data_config):
    dforget = dict()
    rgap, sgap, pcount = calC(row, data_config)

    ## TODO
    dforget["rgaps"], dforget["sgaps"], dforget["pcounts"] = rgap, sgap, pcount
    return dforget


def prepare_data(
    model_name, is_repeat, qidx, dcur, curdforget, dtotal, dforget, t, end, maxlen=200
):
    curqin, curcin, currin, curtin = (
        dcur["curqin"],
        dcur["curcin"],
        dcur["currin"],
        dcur["curtin"],
    )
    cq, cc, cr, ct = dtotal["cq"], dtotal["cc"], dtotal["cr"], dtotal["ct"]
    ## cq: questions, cc: concepts, cr: responses, ct: times

    dqshfts, dcshfts, drshfts, dtshfts, dds, ddshfts = [], [], [], [], dict(), dict()
    dqs, dcs, drs, dts = [], [], [], []
    if model_name == "lpkt":
        curitin = dcur["curitin"]
        cit = dtotal["cit"]
        dits, ditshfts = [], []
    elif model_name == "dimkt":
        cursdin, curqdin = dcur["cursdin"], dcur["curqdin"]
        csd, cqd = dtotal["csd"], dtotal["cqd"]
        dsds, dsdshfts, dqds, dqdshfts = [], [], [], []

    qidxs = []
    qstart = qidx - 1
    for k in range(t, end):
        if is_repeat[k] == 0:
            qstart += 1
            qidxs.append(qstart)
        else:
            qidxs.append(qstart)
        # get start
        start = 0
        cinlen = curcin.shape[1]
        if cinlen >= maxlen - 1:
            start = cinlen - maxlen + 1

        curc, curr = cc.long()[k], cr.long()[k]
        curc, curr = torch.tensor([[curc.item()]]).to(DEVICE), torch.tensor(
            [[curr.item()]]
        ).to(DEVICE)
        dcs.append(curcin[:, start:])
        drs.append(currin[:, start:])

        curc, curr = torch.cat((curcin[:, start + 1 :], curc), axis=1), torch.cat(
            (currin[:, start + 1 :], curr), axis=1
        )
        dcshfts.append(curc)
        drshfts.append(curr)
        if cq.shape[0] > 0:
            curq = cq.long()[k]
            curq = torch.tensor([[curq.item()]]).to(DEVICE)

            dqs.append(curqin[:, start:])
            curq = torch.cat((curqin[:, start + 1 :], curq), axis=1)
            dqshfts.append(curq)
        if ct.shape[0] > 0:
            curt = ct.long()[k]
            curt = torch.tensor([[curt.item()]]).to(DEVICE)

            dts.append(curtin[:, start:])
            curt = torch.cat((curtin[:, start + 1 :], curt), axis=1)
            dtshfts.append(curt)
        if model_name == "lpkt":
            if cit.shape[0] > 0:
                curit = cit.long()[k]
                curit = torch.tensor([[curit.item()]]).to(DEVICE)

                dits.append(curitin[:, start:])
                curit = torch.cat((curitin[:, start + 1 :], curit), axis=1)
                ditshfts.append(curit)
        elif model_name == "dimkt":
            cursd = csd.long()[k]
            cursd = torch.tensor([[cursd.item()]]).to(DEVICE)
            curqd = cqd.long()[k]
            curqd = torch.tensor([[curqd.item()]]).to(DEVICE)

            dsds.append(cursdin[:, start:])
            cursd = torch.cat((cursdin[:, start + 1 :], cursd), axis=1)
            dsdshfts.append(cursd)

            dqds.append(curqdin[:, start:])
            curqd = torch.cat((curqdin[:, start + 1 :], curqd), axis=1)
            dqdshfts.append(curqd)

        d, dshft = dict(), dict()
        if model_name in ["dkt_forget", "bakt_time"]:
            for key in curdforget:
                d[key] = curdforget[key][:, start:]
                dds.setdefault(key, [])
                dds[key].append(d[key])
            for key in dforget:
                curd = torch.tensor([[dforget[key][k]]]).long().to(DEVICE)
                dshft[key] = torch.cat((d[key][:, 1:], curd), axis=1)
                ddshfts.setdefault(key, [])
                ddshfts[key].append(dshft[key])

    finalcs, finalrs = torch.cat(dcs, axis=0), torch.cat(drs, axis=0)
    finalqs, finalqshfts = torch.tensor([]), torch.tensor([])
    finalts, finaltshfts = torch.tensor([]), torch.tensor([])
    if cq.shape[0] > 0:
        finalqs = torch.cat(dqs, axis=0)
        finalqshfts = torch.cat(dqshfts, axis=0)
    if ct.shape[0] > 0:
        finalts = torch.cat(dts, axis=0)
        finaltshfts = torch.cat(dtshfts, axis=0)
    finalcshfts, finalrshfts = torch.cat(dcshfts, axis=0), torch.cat(drshfts, axis=0)
    finald, finaldshft = dict(), dict()
    for key, values in dds.items():
        finald[key] = torch.cat(values, axis=0)
        finaldshft[key] = torch.cat(ddshfts[key], axis=0)

    if model_name == "lpkt":
        finalits, finalitshfts = torch.tensor([]), torch.tensor([])
        if cit.shape[0] > 0:
            finalits = torch.cat(dits, axis=0)
            finalitshfts = torch.cat(ditshfts, axis=0)
    elif model_name == "dimkt":
        finalsds, finalsdshfts, finalqds, finalqdshfts = (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )
        finalsds = torch.cat(dsds, axis=0)
        finalsdshfts = torch.cat(dsdshfts, axis=0)
        finalqds = torch.cat(dqds, axis=0)
        finalqdshfts = torch.cat(dqdshfts, axis=0)

    if model_name == "lpkt":
        return (
            qidxs,
            finalqs,
            finalcs,
            finalrs,
            finalts,
            finalits,
            finalqshfts,
            finalcshfts,
            finalrshfts,
            finaltshfts,
            finalitshfts,
            finald,
            finaldshft,
        )
    elif model_name == "dimkt":
        return (
            qidxs,
            finalqs,
            finalcs,
            finalrs,
            finalts,
            finalqshfts,
            finalcshfts,
            finalrshfts,
            finaltshfts,
            finalsds,
            finalsdshfts,
            finalqds,
            finalqdshfts,
            finald,
            finaldshft,
        )
    else:
        return (
            qidxs,
            finalqs,
            finalcs,
            finalrs,
            finalts,
            finalqshfts,
            finalcshfts,
            finalrshfts,
            finaltshfts,
            finald,
            finaldshft,
        )


def predict_each_group2(
    dtotal: Dict,
    dcur: Dict,
    dforget: Dict,
    curdforget: Dict,
    is_repeat: List[int],
    qidx: int,
    uid: int,
    idx: int,
    model_name: str,
    model: nn.Module,
    t: int,
    end: int,
    fout,
    atkt_pad=False,
    maxlen=200,
):
    """_summary_

    Args:
        dtotal (Dict): {'cq': questions array, 'cc': concepts array,'cr': responses array,'ct': times array}
        dcur (Dict): {'cq': questions array, 'cc': concepts array,'cr': responses array,'ct': times array}
        dforget (Dict): {}
        curdforget (Dict): {}
        is_repeat (List[int]): array check repeate question
        qidx (int): question id
        uid (int): user id
        idx (int): index
        model_name (str): model name
        model (nn.Module): model
        t (int)
        end (int)
        fout (_type_): file writer
        atkt_pad (bool, optional): _description_. Defaults to False.
        maxlen (int, optional): _description_. Defaults to 200.

    Returns:
        _type_: _description_
    """
    ctrues, cpreds = [], []
    if model_name == "lpkt":
        (
            qidxs,
            finalqs,
            finalcs,
            finalrs,
            finalts,
            finalits,
            finalqshfts,
            finalcshfts,
            finalrshfts,
            finaltshfts,
            finalitshfts,
            finald,
            finaldshft,
        ) = prepare_data(
            model_name, is_repeat, qidx, dcur, curdforget, dtotal, dforget, t, end
        )
    elif model_name == "dimkt":
        (
            qidxs,
            finalqs,
            finalcs,
            finalrs,
            finalts,
            finalqshfts,
            finalcshfts,
            finalrshfts,
            finaltshfts,
            finalsds,
            finalsdshfts,
            finalqds,
            finalqdshfts,
            finald,
            finaldshft,
        ) = prepare_data(
            model_name, is_repeat, qidx, dcur, curdforget, dtotal, dforget, t, end
        )
    else:
        (
            qidxs,
            finalqs,
            finalcs,
            finalrs,
            finalts,
            finalqshfts,
            finalcshfts,
            finalrshfts,
            finaltshfts,
            finald,
            finaldshft,
        ) = prepare_data(
            model_name, is_repeat, qidx, dcur, curdforget, dtotal, dforget, t, end
        )
    bidx, bz = 0, 128
    while bidx < finalcs.shape[0]:
        curc, curr = finalcs[bidx : bidx + bz], finalrs[bidx : bidx + bz]
        print(curc)
        curcshft, currshft = (
            finalcshfts[bidx : bidx + bz],
            finalrshfts[bidx : bidx + bz],
        )
        curqidxs = qidxs[bidx : bidx + bz]
        curq, curqshft = torch.tensor([[]]), torch.tensor([[]])
        if finalqs.shape[0] > 0:
            curq = finalqs[bidx : bidx + bz]
            curqshft = finalqshfts[bidx : bidx + bz]
        curt, curtshft = torch.tensor([[]]), torch.tensor([[]])
        if finalts.shape[0] > 0:
            curt = finalts[bidx : bidx + bz]
            curtshft = finaltshfts[bidx : bidx + bz]
        curd, curdshft = dict(), dict()
        if model_name in ["dkt_forget", "bakt_time"]:
            for key in finald:
                curd[key] = finald[key][bidx : bidx + bz]
                curdshft[key] = finaldshft[key][bidx : bidx + bz]
        if model_name == "lpkt":
            curit = finalits[bidx : bidx + bz]
            curitshft = finalitshfts[bidx : bidx + bz]
        if model_name == "dimkt":
            cursd = finalsds[bidx : bidx + bz]
            cursdshft = finalsdshfts[bidx : bidx + bz]
            curqd = finalqds[bidx : bidx + bz]
            curqdshft = finalqdshfts[bidx : bidx + bz]

        ## start predict
        if model_name == "dimkt":
            ccq = curq
            ccc = curc
            ccr = curr
            cct = curt
        else:
            ccq = torch.cat((curq[:, 0:1], curqshft), dim=1)
            ccc = torch.cat((curc[:, 0:1], curcshft), dim=1)
            ccr = torch.cat((curr[:, 0:1], currshft), dim=1)
            print(ccc)
            print(ccr)
            cct = torch.cat((curt[:, 0:1], curtshft), dim=1)
        if model_name in ["dkt_forget", "bakt_time"]:
            dgaps = dict()
            for key, values in curd.items():
                dgaps[key] = values

            for key, values in curdshft.items():
                dgaps["shft_" + key] = values
        if model_name in ["atdkt"]:
            dcurinfos = {"qseqs": curq, "cseqs": curc, "rseqs": curr}
            y = model(dcurinfos)
            y = (y * one_hot(curcshft.long(), model.num_c)).sum(-1)
        elif model_name in ["dkt", "dkt+"]:
            y = model(curc.long(), curr.long())
            y = (y * one_hot(curcshft.long(), model.num_c)).sum(-1)
        elif model_name in ["dkt_forget"]:
            y = model(curc.long(), curr.long(), dgaps)
            y = (y * one_hot(curcshft.long(), model.num_c)).sum(-1)
        elif model_name in ["dkvmn"]:
            y = model(ccc.long(), ccr.long())
            y = y[:, 1:]
        elif model_name in ["kqn", "sakt"]:
            y = model(curc.long(), curr.long(), curcshft.long())
        elif model_name == "lpkt":
            ccit = torch.cat((curit[:, 0:1], curitshft), dim=1)
            y = model(ccq.long(), ccr.long(), ccit.long())
            y = y[:, 1:]
        elif model_name == "dimkt":
            ccsd = cursd
            ccqd = curqd
            y = model(
                ccq.long(),
                ccc.long(),
                ccsd.long(),
                ccqd.long(),
                ccr.long(),
                curqshft.long(),
                curcshft.long(),
                cursdshft.long(),
                curqdshft.long(),
            )

        pred = y[:, -1].tolist()
        true = ccr[:, -1].tolist()

        # save pred res
        ctrues.extend(true)
        cpreds.extend(pred)

        # output
        for i in range(0, curc.shape[0]):
            clist, rlist = curc[i].long().tolist()[0:t], curr[i].long().tolist()[0:t]
            cshftlist, rshftlist = (
                curcshft[i].long().tolist()[0:t],
                currshft[i].long().tolist()[0:t],
            )
            qidx = curqidxs[i]
            predl = 1 if pred[i] >= 0.5 else 0
            fout.write(
                "\t".join(
                    [
                        str(idx),
                        str(uid),
                        str(bidx + i),
                        str(qidx),
                        str(len(clist)),
                        str(clist),
                        str(rlist),
                        str(cshftlist),
                        str(rshftlist),
                        str(true[i]),
                        str(pred[i]),
                        str(predl),
                    ]
                )
                + "\n"
            )

        bidx += bz
    return qidxs, ctrues, cpreds


def predict_each_group(
    dtotal,
    dcur,
    dforget,
    curdforget,
    is_repeat,
    qidx,
    uid,
    idx,
    model_name,
    model,
    t,
    end,
    fout,
    atkt_pad=False,
    maxlen=200,
):
    """use the predict result as next question input"""
    curqin, curcin, currin, curtin = (
        dcur["curqin"],
        dcur["curcin"],
        dcur["currin"],
        dcur["curtin"],
    )
    # print(f"cin8:{curcin}")
    # print(f"rin8:{currin}")
    cq, cc, cr, ct = dtotal["cq"], dtotal["cc"], dtotal["cr"], dtotal["ct"]
    if model_name == "lpkt":
        curitin = dcur["curitin"]
        cit = dtotal["cit"]
    if model_name == "dimkt":
        cursdin = dcur["cursdin"]
        curqdin = dcur["curqdin"]
        csd = dtotal["csd"]
        cqd = dtotal["cqd"]

    nextcin, nextrin = curcin, currin
    import copy

    nextdforget = copy.deepcopy(curdforget)
    ctrues, cpreds = [], []
    for k in range(t, end):
        qin, cin, rin, tin = curqin, curcin, currin, curtin
        # print(f"cin9:{cin}")
        # print(f"rin9:{rin}")
        if model_name == "lpkt":
            itin = curitin
        if model_name == "dimkt":
            sdin = cursdin
            qdin = curqdin
        # print("cin: ", cin)
        start = 0
        cinlen = cin.shape[1]
        if cinlen >= maxlen - 1:
            start = cinlen - maxlen + 1

        cin, rin = cin[:, start:], rin[:, start:]
        # print(f"cin10:{cin}")
        # print(f"rin10:{rin}")

        if cq.shape[0] > 0:
            qin = qin[:, start:]
        if ct.shape[0] > 0:
            tin = tin[:, start:]
        if model_name == "lpkt":
            itin = itin[:, start:]
        if model_name == "dimkt":
            sdin = sdin[:, start:]
            qdin = qdin[:, start:]
        # print(f"start: {start}, cin: {cin.shape}")
        cout, true = cc.long()[k], cr.long()[k]
        qout = None if cq.shape[0] == 0 else cq.long()[k]
        tout = None if ct.shape[0] == 0 else ct.long()[k]

        if model_name == "lpkt":
            itout = None if cit.shape[0] == 0 else cit.long()[k]
        if model_name == "dimkt":
            sdout = None if csd.shape[0] == 0 else csd.long()[k]
            qdout = None if cqd.shape[0] == 0 else cqd.long()[k]
        if model_name in ["dkt", "dkt+"]:
            y = model(cin.long(), rin.long())
            # print(y)
            pred = y[0][-1][cout.item()]
        if model_name in ["dkt_forget", "bakt_time"]:
            din = dict()
            for key in curdforget:
                din[key] = curdforget[key][:, start:]
            dcur = dict()
            for key in dforget:
                curd = torch.tensor([[dforget[key][k]]]).long().to(DEVICE)
                dcur[key] = torch.cat((din[key][:, 1:], curd), axis=1)
            dgaps = dict()
            for key, values in din.items():
                dgaps[key] = values
            for key, values in dcur.items():
                dgaps["shft_" + key] = values
        if model_name in ["atdkt"]:  ## need change!
            # create input
            dcurinfos = {"qseqs": qin, "cseqs": cin, "rseqs": rin}
            y = model(dcurinfos)
            pred = y[0][-1][cout.item()]
        elif model_name in ["dkt", "dkt+"]:
            y = model(cin.long(), rin.long())
            # print(y)
            pred = y[0][-1][cout.item()]
        elif model_name == "dkt_forget":
            # y = model(cin.long(), rin.long(), din, dcur)
            y = model(cin.long(), rin.long(), dgaps)
            pred = y[0][-1][cout.item()]
        elif model_name in ["kqn", "sakt"]:
            curc = torch.tensor([[cout.item()]]).to(DEVICE)
            cshft = torch.cat((cin[:, 1:], curc), axis=1)
            y = model(cin.long(), rin.long(), cshft.long())
            pred = y[0][-1]
        elif model_name == "saint":
            if qout is not None:
                curq = torch.tensor([[qout.item()]]).to(DEVICE)
                qin = torch.cat((qin, curq), axis=1)
            curc = torch.tensor([[cout.item()]]).to(DEVICE)
            cin = torch.cat((cin, curc), axis=1)

            y = model(qin.long(), cin.long(), rin.long())
            pred = y[0][-1]
        elif model_name in ["atkt", "atktfix"]:
            if atkt_pad:
                oricinlen = cin.shape[1]
                padlen = maxlen - 1 - oricinlen
                # print(f"padlen: {padlen}, cin: {cin.shape}")
                pad = torch.tensor([0] * (padlen)).unsqueeze(0).to(DEVICE)
                # curc = torch.tensor([[cout.item()]]).to(device)
                # cshft = torch.cat((cin[:,1:],curc), axis=1)
                cin = torch.cat((cin, pad), axis=1)
                rin = torch.cat((rin, pad), axis=1)
            y, _ = model(cin.long(), rin.long())
            # print(f"y: {y}")
            if atkt_pad:
                # print(f"use idx: {oricinlen-1}")
                pred = y[0][oricinlen - 1][cout.item()]
            else:
                pred = y[0][-1][cout.item()]
        elif model_name in ["dkvmn", "deep_irt", "skvmn"]:
            curc, curr = torch.tensor([[cout.item()]]).to(DEVICE), torch.tensor(
                [[true.item()]]
            ).to(DEVICE)
            cin, rin = torch.cat((cin, curc), axis=1), torch.cat((rin, curr), axis=1)
            # print(f"cin: {cin.shape}, curc: {curc.shape}")
            y = model(cin.long(), rin.long())
            pred = y[0][-1]
        elif model_name in [
            "akt",
            "akt_vector",
            "akt_norasch",
            "akt_mono",
            "akt_attn",
            "aktattn_pos",
            "aktmono_pos",
            "akt_raschx",
            "akt_raschy",
            "aktvec_raschx",
        ]:
            if qout is not None:
                curq = torch.tensor([[qout.item()]]).to(DEVICE)
                qin = torch.cat((qin, curq), axis=1)
            curc, curr = torch.tensor([[cout.item()]]).to(DEVICE), torch.tensor(
                [[1]]
            ).to(DEVICE)
            cin, rin = torch.cat((cin, curc), axis=1), torch.cat((rin, curr), axis=1)

            y, reg_loss = model(cin.long(), rin.long(), qin.long())
            pred = y[0][-1]
        elif model_name in ["bakt_time"]:
            if qout is not None:
                curq = torch.tensor([[qout.item()]]).to(DEVICE)
                qinshft = torch.cat((qin[:, 1:], curq), axis=1)
            else:
                qin = torch.tensor([[]]).to(DEVICE)
                qinshft = torch.tensor([[]]).to(DEVICE)
            curc, curr = torch.tensor([[cout.item()]]).to(DEVICE), torch.tensor(
                [[1]]
            ).to(DEVICE)
            cinshft, rinshft = torch.cat((cin[:, 1:], curc), axis=1), torch.cat(
                (rin[:, 1:], curr), axis=1
            )
            dcurinfos = {
                "qseqs": qin,
                "cseqs": cin,
                "rseqs": rin,
                "shft_qseqs": qinshft,
                "shft_cseqs": cinshft,
                "shft_rseqs": rinshft,
            }

            y = model(dcurinfos, dgaps)
            pred = y[0][-1]
        elif model_name in ["simplekt", "sparsekt"]:
            if qout is not None:
                curq = torch.tensor([[qout.item()]]).to(DEVICE)
                qinshft = torch.cat((qin[:, 1:], curq), axis=1)
            else:
                qin = torch.tensor([[]]).to(DEVICE)
                qinshft = torch.tensor([[]]).to(DEVICE)
            curc, curr = torch.tensor([[cout.item()]]).to(DEVICE), torch.tensor(
                [[1]]
            ).to(DEVICE)
            cinshft, rinshft = torch.cat((cin[:, 1:], curc), axis=1), torch.cat(
                (rin[:, 1:], curr), axis=1
            )
            dcurinfos = {
                "qseqs": qin,
                "cseqs": cin,
                "rseqs": rin,
                "shft_qseqs": qinshft,
                "shft_cseqs": cinshft,
                "shft_rseqs": rinshft,
            }

            y = model(dcurinfos)
            pred = y[0][-1]
        elif model_name == "lpkt":
            if itout is not None:
                curit = torch.tensor([[itout.item()]]).to(DEVICE)
                itin = torch.cat((itin, curit), axis=1)
            curq, curr = torch.tensor([[qout.item()]]).to(DEVICE), torch.tensor(
                [[1]]
            ).to(DEVICE)
            # curc, curr = torch.tensor([[cout.item()]]).to(device), torch.tensor([[true.item()]]).to(device)
            qin, rin = torch.cat((qin, curq), axis=1), torch.cat((rin, curr), axis=1)
            y = model(qin.long(), rin.long(), itin.long())
            # print(f"pred: {y}")
            # label = [1 if x >= 0.5 else 0 for x in y[0]]
            # print(f"pred_labels: {label}")
            pred = y[0][-1]
        elif model_name == "dimkt":
            if sdout is not None and qdout is not None:
                cursd = torch.tensor([[sdout.item()]]).to(DEVICE)
                # sdin = torch.cat((sdin, cursd), axis=1)
                curqd = torch.tensor([[qdout.item()]]).to(DEVICE)
                # qdin = torch.cat((qdin, curqd), axis=1)
            curq, curc, curr = (
                torch.tensor([[qout.item()]]).to(DEVICE),
                torch.tensor([[cout.item()]]).to(DEVICE),
                torch.tensor([[1]]).to(DEVICE),
            )
            # qin, cin, rin = torch.cat((qin, curq), axis=1), torch.cat((cin, curc), axis=1), torch.cat((rin, curr), axis=1)
            qinshft, cinshft, rinshft, sdinshft, qdinshft = (
                torch.cat((qin[:, 1:], curq), axis=1),
                torch.cat((cin[:, 1:], curc), axis=1),
                torch.cat((rin[:, 1:], curr), axis=1),
                torch.cat((sdin[:, 1:], cursd), axis=1),
                torch.cat((qdin[:, 1:], curqd), axis=1),
            )
            # qinshft, cinshft, rinshft, sdinshft, qdinshft = qin[:,1:], cin[:,1:], rin, sdin[:,1:], qdin[:,1:]
            # print(f"qin:{qin.shape}, qinshft:{qinshft.shape}")
            y = model(
                qin.long(),
                cin.long(),
                sdin.long(),
                qdin.long(),
                rin.long(),
                qinshft.long(),
                cinshft.long(),
                sdinshft.long(),
                qdinshft.long(),
            )
            pred = y[0][-1]
        elif model_name == "gkt":
            curc, curr = torch.tensor([[cout.item()]]).to(DEVICE), torch.tensor(
                [[1]]
            ).to(DEVICE)
            cin, rin = torch.cat((cin, curc), axis=1), torch.cat((rin, curr), axis=1)
            y = model(cin.long(), rin.long())
            # print(f"y.shape is {y.shape},cin shape is {cin.shape}")
            pred = y[0][-1]
        elif model_name == "hawkes":
            curc, curr = torch.tensor([[cout.item()]]).to(DEVICE), torch.tensor(
                [[1]]
            ).to(DEVICE)
            if tout is not None:
                curt = torch.tensor([[tout.item()]]).to(DEVICE)
                tin = torch.cat((tin, curt), axis=1)
            else:
                tin = torch.tensor([[]]).to(DEVICE)
            if qout is not None:
                curq = torch.tensor([[qout.item()]]).to(DEVICE)
                qin = torch.cat((qin, curq), axis=1)
            curc, curr = torch.tensor([[cout.item()]]).to(DEVICE), torch.tensor(
                [[1]]
            ).to(DEVICE)
            cin, rin = torch.cat((cin, curc), axis=1), torch.cat((rin, curr), axis=1)
            # print(f"cin: {cin.shape}, qin: {qin.shape}, tin: {tin.shape}, rin: {rin.shape}")
            y = model(cin.long(), qin.long(), tin.long(), rin.long())
            pred = y[0][-1]

        predl = 1 if pred.item() >= 0.5 else 0
        cpred = torch.tensor([[predl]]).to(DEVICE)

        nextqin = cq[0 : k + 1].unsqueeze(0) if cq.shape[0] > 0 else qin
        nexttin = ct[0 : k + 1].unsqueeze(0) if ct.shape[0] > 0 else tin
        nextcin = cc[0 : k + 1].unsqueeze(0)
        nextrin = torch.cat((nextrin, cpred), axis=1)  ### change!!
        if model_name == "lpkt":
            nexttin = ct[0 : k + 1].unsqueeze(0) if ct.shape[0] > 0 else tin
            nextitin = cit[0 : k + 1].unsqueeze(0) if cit.shape[0] > 0 else itin
        if model_name == "dimkt":
            nextsdin = csd[0 : k + 1].unsqueeze(0) if csd.shape[0] > 0 else sdin
            nextqdin = cqd[0 : k + 1].unsqueeze(0) if cqd.shape[0] > 0 else qdin
        # update nextdforget
        if model_name in ["dkt_forget", "bakt_time"]:
            for key in nextdforget:
                curd = torch.tensor([[dforget[key][k]]]).long().to(DEVICE)
                nextdforget[key] = torch.cat((nextdforget[key], curd), axis=1)
        # print(f"bz: {bz}, t: {t}, pred: {pred}, true: {true}")

        # save pred res
        ctrues.append(true.item())
        cpreds.append(pred.item())

        # output
        clist, rlist = (
            cin.squeeze(0).long().tolist()[0:k],
            rin.squeeze(0).long().tolist()[0:k],
        )
        # print("\t".join([str(idx), str(uid), str(k), str(qidx), str(is_repeat[t:end]), str(len(clist)), str(clist), str(rlist), str(cout.item()), str(true.item()), str(pred.item()), str(predl)]))
        fout.write(
            "\t".join(
                [
                    str(idx),
                    str(uid),
                    str(k),
                    str(qidx),
                    str(is_repeat[t:end]),
                    str(len(clist)),
                    str(clist),
                    str(rlist),
                    str(cout.item()),
                    str(true.item()),
                    str(pred.item()),
                    str(predl),
                ]
            )
            + "\n"
        )
    # nextcin, nextrin = nextcin.unsqueeze(0), nextrin.unsqueeze(0)
    if model_name == "lpkt":
        return nextqin, nextcin, nextrin, nexttin, nextitin, nextdforget, ctrues, cpreds
    elif model_name == "dimkt":
        return nextqin, nextcin, nextrin, nexttin, nextsdin, nextqdin, ctrues, cpreds
    else:
        return nextqin, nextcin, nextrin, nexttin, nextdforget, ctrues, cpreds


def evaluate_splitpred_question(
    model,
    data_config,
    testf,
    model_name,
    save_path="",
    use_pred=False,
    train_ratio=0.2,
    atkt_pad=False,
):
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")

    with torch.no_grad():
        idx = 0
        df = pd.read_csv(testf)
        dcres, dqres = {"trues": [], "preds": []}, {
            "trues": [],
            "late_mean": [],
            "late_vote": [],
            "late_all": [],
        }
        for _, row in df.iterrows():
            model.eval()
            dforget = (
                dict()
                if model_name not in ["dkt_forget", "bakt_time"]
                else get_info_dkt_forget(row, data_config)
            )

            concepts, responses = row["concepts"].split(","), row["responses"].split(
                ","
            )  # lấy concepts danh sách câu hỏi và responses của người dùng

            rs = []
            for item in responses:
                newr = item if item != "-1" else "0"  # default -1 to 0
                rs.append(newr)
            responses = rs

            curl = len(responses)
            is_repeat = (
                ["0"] * curl if "is_repeat" not in row else row["is_repeat"].split(",")
            )  # nếu không có cột is_repeate trong df thì is_repeat là mảng 0
            is_repeat = [int(s) for s in is_repeat]

            questions = (
                [] if "questions" not in row else row["questions"].split(",")
            )  # lấy danh sách câu hỏi
            times = (
                [] if "timestamps" not in row else row["timestamps"].split(",")
            )  # danh sách thời gian phản hồi của người dùng

            if model_name == "dimkt":
                sds = {}
                qds = {}
                with open(
                    os.path.join(
                        data_config["dpath"],
                        f"skills_difficult_{model.difficult_levels}.csv",
                    ),
                    "r",
                    encoding="utf-8",
                ) as file:
                    reader = csv.reader(file)
                    sds_keys = next(reader)
                    sds_vals = next(reader)
                    for i, sds_key in enumerate(sds_keys):
                        sds[int(sds_key)] = int(sds_vals[i])

                with open(
                    os.path.join(
                        data_config["dpath"],
                        f"questions_difficult_{model.difficult_levels}.csv",
                    ),
                    "r",
                    encoding="utf-8",
                ) as file:
                    reader = csv.reader(file)
                    qds_keys = next(reader)
                    qds_vals = next(reader)
                    for i, qds_key in enumerate(qds_keys):
                        qds[int(qds_key)] = int(qds_vals[i])
                sds_keys = [int(_) for _ in sds_keys]
                qds_keys = [int(_) for _ in qds_keys]

                seq_sds, seq_qds = [], []
                temp = [int(_) for _ in row["concepts"].split(",")]
                for j in temp:
                    if j == -1:
                        seq_sds.append(-1)
                    elif j not in sds_keys:
                        seq_sds.append(1)
                    else:
                        seq_sds.append(int(sds[j]))

                temp = [int(_) for _ in row["questions"].split(",")]
                for j in temp:
                    if j == -1:
                        seq_qds.append(-1)
                    elif j not in qds_keys:
                        seq_qds.append(1)
                    else:
                        seq_qds.append(int(qds[j]))

            qlen, qtrainlen, ctrainlen = get_cur_teststart(is_repeat, train_ratio)
            cq = torch.tensor([int(s) for s in questions]).to(DEVICE)
            cc = torch.tensor([int(s) for s in concepts]).to(DEVICE)
            cr = torch.tensor([int(s) for s in responses]).to(DEVICE)
            ct = torch.tensor([int(s) for s in times]).to(DEVICE)
            dtotal = {"cq": cq, "cc": cc, "cr": cr, "ct": ct}

            curcin, currin = cc[0:ctrainlen].unsqueeze(0), cr[0:ctrainlen].unsqueeze(0)

            curqin = cq[0:ctrainlen].unsqueeze(0) if cq.shape[0] > 0 else cq
            curtin = ct[0:ctrainlen].unsqueeze(0) if ct.shape[0] > 0 else ct
            if model_name == "dimkt":
                csd = torch.tensor(seq_sds).to(DEVICE)
                cqd = torch.tensor(seq_qds).to(DEVICE)
                dtotal["csd"] = csd
                dtotal["cqd"] = cqd
                cursdin = csd[0:ctrainlen].unsqueeze(0) if csd.shape[0] > 0 else csd
                curqdin = cqd[0:ctrainlen].unsqueeze(0) if cqd.shape[0] > 0 else cqd
            dcur = {
                "curqin": curqin,
                "curcin": curcin,
                "currin": currin,
                "curtin": curtin,
            }
            if model_name == "dimkt":
                dcur["cursdin"] = cursdin
                dcur["curqdin"] = curqdin

            curdforget = dict()
            for key, values in dforget.items():
                dforget[key] = torch.tensor(values).to(DEVICE)
                curdforget[key] = values[0:ctrainlen].unsqueeze(0)

            t = ctrainlen

            if not use_pred:
                uid, end = row["uid"], curl
                qidx = qtrainlen
                qidxs, ctrues, cpreds = predict_each_group2(
                    dtotal,
                    dcur,
                    dforget,
                    curdforget,
                    is_repeat,
                    qidx,
                    uid,
                    idx,
                    model_name,
                    model,
                    t,
                    end,
                    fout,
                    atkt_pad,
                )
                save_currow_question_res(
                    idx, dcres, dqres, qidxs, ctrues, cpreds, uid, fout
                )
            else:
                qidx = qtrainlen
                while t < curl:
                    rtmp = [t]
                    for k in range(t + 1, curl):
                        if is_repeat[k] != 0:
                            rtmp.append(k)
                        else:
                            break

                    end = rtmp[-1] + 1
                    uid = row["uid"]
                    if model_name == "dimkt":
                        (
                            curqin,
                            curcin,
                            currin,
                            curtin,
                            cursdin,
                            curqdin,
                            ctrues,
                            cpreds,
                        ) = predict_each_group(
                            dtotal,
                            dcur,
                            dforget,
                            curdforget,
                            is_repeat,
                            qidx,
                            uid,
                            idx,
                            model_name,
                            model,
                            t,
                            end,
                            fout,
                            atkt_pad,
                        )
                        dcur = {
                            "curqin": curqin,
                            "curcin": curcin,
                            "currin": currin,
                            "curtin": curtin,
                            "cursdin": cursdin,
                            "curqdin": curqdin,
                        }
                    else:
                        (
                            curqin,
                            curcin,
                            currin,
                            curtin,
                            curdforget,
                            ctrues,
                            cpreds,
                        ) = predict_each_group(
                            dtotal,
                            dcur,
                            dforget,
                            curdforget,
                            is_repeat,
                            qidx,
                            uid,
                            idx,
                            model_name,
                            model,
                            t,
                            end,
                            fout,
                            atkt_pad,
                        )
                        dcur = {
                            "curqin": curqin,
                            "curcin": curcin,
                            "currin": currin,
                            "curtin": curtin,
                        }

                    late_mean, late_vote, late_all = save_each_question_res(
                        dcres, dqres, ctrues, cpreds
                    )
                    fout.write(
                        "\t".join(
                            [
                                str(idx),
                                str(uid),
                                str(qidx),
                                str(late_mean),
                                str(late_vote),
                                str(late_all),
                            ]
                        )
                        + "\n"
                    )
                    t = end
                    qidx += 1
                    fout.write("\n")
            idx += 1

        try:
            dfinal = cal_predres(dcres, dqres)
            for key, values in dfinal.items():
                fout.write(key + "\t" + str(values) + "\n")
        except:
            print(f"can't output auc and accuracy!")
            dfinal = dict()
    return dfinal
