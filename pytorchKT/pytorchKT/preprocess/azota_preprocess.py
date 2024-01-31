import pandas as pd

from utils import write_txt


KEY = ["user_id", "attribute_value"]


def get_data(df: pd.DataFrame, keys: list, stares: list):
    uids = df[keys[0]].unique()
    cids = df[keys[1]].unique()

    avgins = round(df.shape[0] / len(uids), 4)
    ins, users, questions, concepts = df.shape[0], len(uids), "NA", len(cids)
    avgcqf, naf = "NA", "NA"
    curr = [ins, users, questions, concepts, avgins, avgcqf, naf]
    stares.append(",".join([str(s) for s in curr]))
    return ins, users, questions, concepts, avgins, avgcqf, naf


def read_data_from_csv(read_file, write_file):
    stares = []
    df = pd.read_csv(read_file)

    ins, users, questions, concepts, avgins, avgcq, na = get_data(
        df=df, keys=KEY, stares=stares
    )
    print(
        f"after drop interaction num: {ins}, user num: {users}, question num: {questions}, concept num: {concepts}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}"
    )

    ui_df = df.groupby("user_id")
    user_inters = []

    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["exam_id"])
        seq_len = len(tmp_inter)
        seq_skills = tmp_inter["attribute_value"].astype(str)
        seq_ans = tmp_inter["correct"].astype(str)
        seq_problems = tmp_inter["question_id"].astype(str)
        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]

        assert seq_len == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [
                [str(user), str(seq_len)],
                seq_problems,
                seq_skills,
                seq_ans,
                seq_start_time,
                seq_response_cost,
            ]
        )

    write_txt(write_file, user_inters)
    print("\n".join(stares))
    return
