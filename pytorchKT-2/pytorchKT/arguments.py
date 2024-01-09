import argparse


def get_eval_argments():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "eval"])
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--test_filename", type=str, default="test.csv")
    parser.add_argument("--use_pred", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--atkt_pad", type=int, default=0)

    return parser.parse_args()


def get_train_argments():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "eval"])
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="assist2015",
        choices=["assist2015", "assist2009", "ednet", "junyi"],
    )
    parser.add_argument(
        "--model_name", type=str, default="dkt", choices=["dkt", "sakt", "dkvmn"]
    )
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--dim_s", type=int, default=200)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--num_en", type=int, default=1)
    parser.add_argument("--size_m", type=int, default=50)

    return parser.parse_args()
