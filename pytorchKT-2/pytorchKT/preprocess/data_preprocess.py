import os, sys
import argparse
from split_datasets import split_concept
from split_datasets_que import split_question

dname2paths = {
    "assist2009": "/home/toan/d/Azota/pytorchKnowledtracing/pytorchKT-2/dataset/ASSISTments2009/skill_builder_data_corrected_collapsed.csv",
    "assist2015": "/home/toan/d/Azota/pytorchKnowledtracing/pytorchKT-2/dataset/ASSISTments2015/2015_100_skill_builders_main_problems.csv",
}
config = "/home/toan/d/Azota/pytorchKnowledtracing/pytorchKT-2/pytorchKT/configs/data_config.json"


def process_raw_data(dataset_name, dname2paths):
    readf = dname2paths[dataset_name]
    dname = "/".join(readf.split("/")[0:-1])
    writef = os.path.join(dname, "data.txt")
    print(f"Start preprocessing data: {dataset_name}")
    if dataset_name == "assist2009":
        from assist2009_preprocess import read_data_from_csv
    elif dataset_name == "assist2015":
        from assist2015_preprocess import read_data_from_csv

    # metap = os.path.join(dname, "metadata")
    read_data_from_csv(readf, writef)

    return dname, writef


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="assist2009")
    parser.add_argument("-m", "--min_seq_len", type=int, default=3)
    parser.add_argument("-l", "--maxlen", type=int, default=70)
    parser.add_argument("-k", "--kfold", type=int, default=5)
    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    args = parser.parse_args()

    print(args)
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print("-" * 50)
    print(f"dname: {dname}, writef: {writef}")
    # split
    os.system("rm " + dname + "/*.pkl")

    # for concept level model
    split_concept(
        dname,
        writef,
        args.dataset_name,
        config,
        args.min_seq_len,
        args.maxlen,
        args.kfold,
    )
    print("=" * 100)

    # for question level model
    split_question(
        dname,
        writef,
        args.dataset_name,
        config,
        args.min_seq_len,
        args.maxlen,
        args.kfold,
    )
