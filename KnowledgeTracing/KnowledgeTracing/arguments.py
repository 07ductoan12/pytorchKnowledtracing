from argparse import ArgumentParser


def get_train_argment():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkt+, dkvmn, sakt, gkt]. \
            The default model is dkt.",
        # choices=["dkt", "dkt+", "dkvmn", "sakt", "gkt"],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ASSIST2009",
        help="The name of the dataset to use in training. \
            The possible datasets are in \
            [ASSIST2009, ASSIST2015, Algebra2005, Statics2011]. \
            The default dataset is ASSIST2009.",
        # choices=["ASSIST2009", "ASSIST2015", "Algebra2005", "Statics2011"],
    )
    return parser.parse_args()
