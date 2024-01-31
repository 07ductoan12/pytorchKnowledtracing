import argparse
import sys


def get_eval_argments():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "eval"])
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--test_filename", type=str, default="test.csv")
    parser.add_argument("--use_pred", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--atkt_pad", type=int, default=0)

    return parser.parse_args()


def get_train_arguments():
    model_name = sys.argv[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "eval"])
    parser.add_argument(
        "model_name",
        type=str,
        default="dimkt_cc",
        choices=[
            "dkt",
            "sakt",
            "dkvmn",
            "dimkt",
            "dkt+",
            "dimkt_cc",
            "deep_irt",
            "hawkes",
            "skvmn",
        ],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="assist2009",
        choices=["assist2015", "assist2009", "ednet", "junyi", "azota"],
    )
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    if model_name in ["dimkt", "dimkt_cc"]:
        parser.add_argument("--emb_size", type=int, default=512)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_steps", type=int, default=199)
        parser.add_argument("--difficult_levels", type=int, default=100)
    elif model_name == "dkt":
        parser.add_argument("--emb_size", type=int, default=200)
    elif model_name == "dkvmn" or model_name == "skvmn":
        parser.add_argument("--dim_s", type=int, default=200)
        parser.add_argument("--size_m", type=int, default=50)
    elif model_name == "dkt+":
        parser.add_argument("--emb_size", type=int, default=200)
        parser.add_argument("--lambda_r", type=float, default=0.01)
        parser.add_argument("--lambda_w1", type=float, default=0.003)
        parser.add_argument("--lambda_w2", type=float, default=3.0)
    elif model_name == "sakt":
        parser.add_argument("--emb_size", type=int, default=256)
        parser.add_argument("--num_attn_heads", type=int, default=8)
        parser.add_argument("--num_en", type=int, default=1)
    elif model_name == "deep_irt":
        parser.add_argument("--dim_s", type=int, default=200)
        parser.add_argument("--size_m", type=int, default=50)
        parser.add_argument("--emb_size", type=int, default=256)
    elif model_name == "hawkes":
        parser.add_argument("--emb_size", type=int, default=64)
        parser.add_argument("--time_log", type=int, default=5)
        parser.add_argument("--l2", type=float, default=1e-5)
    else:
        exit()

    return parser.parse_args()
