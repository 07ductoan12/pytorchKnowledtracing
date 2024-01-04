from KnowledgeTracing import Trainer, Arguments

import sys
import os

if __name__ == "__main__":
    if sys.argv[1] == "train":
        print("Training .................")
        # args = Arguments.get_train_argment()

        # model_name = args.model_name
        # dataset_name = args.dataset_name
        model_name = "dkt+"
        dataset_name = "ASSIST2009"
        Trainer.train(model_name, dataset_name)

    else:
        print(f"{sys.argv[1]} not found")
