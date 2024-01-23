import os
import csv
import torch
import pandas as pd
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_each_kc(dcur, model_name, model):
    curc, curr = dcur["curc"], dcur["curr"]
    curcshft, currshft = dcur["curcshft"], dcur["currshft"]

    if model_name == "dimkt":
        ccq = dcur["curq"]
        ccc = curc
        ccr = curr
        cursd = dcur["cursd"]
        cursdshft = dcur["cursdshft"]
        curqd = dcur["curqd"]
        curqdshft = dcur["curqdshft"]
        curqshft = dcur["curqshft"]

    else:
        ccc = torch.cat((curc[:, 0:1], curcshft), dim=1)
        ccr = torch.cat((curr[:, 0:1], currshft), dim=1)

    if model_name in ["dkt", "dkt+"]:
        y = model(curc.long(), curr.long())
        y = (y * one_hot(curcshft.long(), model.num_c)).sum(-1)
    elif model_name == "dkvmn":
        y = model(ccc.long(), ccr.long())
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
    return pred[0]


def eval_kcs_one_user(
    model,
    concepts,
    responses,
    model_name,
    questions: list | None = None,
    seq_sds: dict | None = None,
    seq_qds: dict | None = None,
):
    with torch.no_grad():
        model.eval()
        rs = [item if item != "-1" else "0" for item in responses]
        responses = rs

        cc, cr = torch.tensor([int(s) for s in concepts]).to(DEVICE), torch.tensor(
            [int(r) for r in responses]
        ).to(DEVICE)

        if model_name == "dimkt":
            cq = torch.tensor([int(s) for s in questions]).to(DEVICE)
            csd = torch.tensor(seq_sds).to(DEVICE)
            cqd = torch.tensor(seq_qds).to(DEVICE)

        unique_cc = list(set(cc.tolist()))
        results = []

        for concept in unique_cc:
            if (cc == concept).nonzero(as_tuple=True)[-1].shape[0] < 3:
                continue

            index = int((cc == concept).nonzero(as_tuple=True)[-1][-1])
            start = max(0, index - 200)

            dcur = {
                "curc": cc[start:index].unsqueeze(dim=0),
                "curr": cr[start:index].unsqueeze(dim=0),
                "curcshft": cc[start + 1 : index + 1].unsqueeze(dim=0),
                "currshft": cr[start + 1 : index + 1].unsqueeze(dim=0),
            }

            if model_name == "dimkt":
                dcur["curq"] = cq[start:index].unsqueeze(dim=0)
                dcur["curqshft"] = cq[start + 1 : index + 1].unsqueeze(dim=0)
                dcur["cursd"] = csd[start:index].unsqueeze(dim=0)
                dcur["curqd"] = cqd[start:index].unsqueeze(dim=0)
                dcur["cursdshft"] = csd[start + 1 : index + 1].unsqueeze(dim=0)
                dcur["curqdshft"] = cqd[start + 1 : index + 1].unsqueeze(dim=0)

            pred = predict_each_kc(dcur, model_name, model)

            result = {
                "id_concept": str(dcur["curcshft"].squeeze().long().tolist()[-1]),
                "len_interactions": str(len(dcur["curc"].squeeze().long().tolist())),
                "concept_list": str(dcur["curc"].squeeze().long().tolist()),
                "response_list": str(dcur["curr"].squeeze().long().tolist()),
                "concept_shftlist": str(dcur["curcshft"].squeeze().long().tolist()),
                "response_shftlist": str(dcur["currshft"].squeeze().long().tolist()),
                "pred": str(pred),
            }
            results.append(result)
    return results


def eval_kcs(model, data_config, testf, model_name, save_path=""):
    fout = open(save_path, "w", encoding="utf-8") if save_path else None

    idx = 0
    df = pd.read_csv(testf)

    for _, row in df.iterrows():
        concepts, responses = row["concepts"].split(","), row["responses"].split(",")
        questions = [] if "questions" not in row else row["questions"].split(",")

        rs = [item if item != "-1" else "0" for item in responses]
        responses = rs

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
                    qds[int(qds_key)] = qds_vals[i]

            sds_keys = [int(_) for _ in sds_keys]
            sds_vals = [int(_) for _ in sds_vals]

            seq_sds, seq_qds = [], []

            for concept in concepts:
                concept = int(concept)
                if concept == -1:
                    seq_sds.append(-1)
                elif concept not in sds_keys:
                    seq_sds.append(1)
                else:
                    seq_sds.append(int(sds[concept]))

            for question in questions:
                question = int(question)
                if question == -1:
                    seq_qds.append(-1)
                elif question not in qds_keys:
                    seq_qds.append(1)
                else:
                    seq_qds.append(int(qds[question]))

        uid = row["uid"]

        results = eval_kcs_one_user(
            model,
            concepts,
            responses,
            questions,
            model_name,
            seq_sds,
            seq_qds,
        )

        pred_concepts = []
        unique_cc = []
        for result in results:
            pred_concepts.append(float(result["pred"]))
            unique_cc.append(int(result["id_concept"]))
            if fout:
                fout.write(
                    "\t".join(
                        [
                            str(idx),
                            str(uid),
                            result["id_concept"],
                            result["len_interactions"],
                            result["concept_list"],
                            result["response_list"],
                            result["concept_shftlist"],
                            result["response_shftlist"],
                            result["pred"],
                        ]
                    )
                    + "\n"
                )

        if fout:
            for pred, concept in zip(pred_concepts, unique_cc):
                if pred != "nan":
                    fout.write(
                        f"Concept: {concept}" + " " * 10 + f"model pred: {pred}" + "\n"
                    )

            fout.write("\n")

        idx += 1

    if fout:
        fout.close()
