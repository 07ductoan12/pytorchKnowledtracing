from pytorchKT.config import ARGS
import pytorchKT.utils as utils
from pytorchKT.preprocess import UserSepDataset
from pytorchKT.models import DKT
from pytorchKT.contanst import QUESTION_NUM
from pytorchKT.trainner import Trainer
import numpy as np
import os


def get_model():
    # if ARGS.model == "DKT":
    model = DKT(
        ARGS.input_dim,
        ARGS.hidden_dim,
        ARGS.num_layers,
        QUESTION_NUM[ARGS.dataset_name],
        ARGS.dropout,
    ).to(ARGS.device)
    d_model = ARGS.hidden_dim

    return model, d_model


def run(i):
    user_base_path = os.path.join(ARGS.base_path, ARGS.dataset_name)
    train_data_path = os.path.join(user_base_path, str(i), "train")
    val_data_path = f"{user_base_path}/{i}/val/"
    test_data_path = f"{user_base_path}/{i}/test/"

    train_sample_infos, num_of_train_user = utils.get_data_user_sep(train_data_path)
    val_sample_infos, num_of_val_user = utils.get_data_user_sep(val_data_path)
    test_sample_infos, num_of_test_user = utils.get_data_user_sep(test_data_path)

    train_data = UserSepDataset("train", train_sample_infos, ARGS.dataset_name)
    val_data = UserSepDataset("val", val_sample_infos, ARGS.dataset_name)
    test_data = UserSepDataset("test", test_sample_infos, ARGS.dataset_name)

    print(
        f"Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_infos)}"
    )
    print(
        f"Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_infos)}"
    )
    print(
        f"Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}"
    )
    model, d_model = get_model()

    trainer = Trainer(
        model=model,
        device=ARGS.device,
        warm_up_step_count=ARGS.warm_up_step_count,
        d_model=d_model,
        num_epochs=ARGS.num_epochs,
        weight_path=ARGS.weight_path,
        lr=ARGS.lr,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )

    trainer.train()
    trainer.test(0)
    return trainer.test_acc, trainer.test_auc


if __name__ == "__main__":
    test_acc, test_auc = run(1)
