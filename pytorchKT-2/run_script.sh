#!/bin/bash

SCRIPT_PATH="./main.py"

set_model_args() {
    COMMON_ARGS=(
        "--save_dir" "saved_model"
        "--dataset_name" "$DATASET_NAME"
        "--emb_type" "qid"
        "--seed" "42"
        "--fold" "0"
        "--dropout" "0.2"
        "--learning_rate" "1e-3"
    )
    if [ "$MODEL_NAME" == "dimkt" ]; then
        MODEL_ARGS=(
            "--emb_size" "256"
            "--difficult_levels" "100"
            "--num_steps" "199"
        )
    elif [ "$MODEL_NAME" == "dkt" ]; then
        MODEL_ARGS=(
            "--emb_size" "256"
        )
    elif [ "$MODEL_NAME" == "dkvmn" ]; then
        MODEL_ARGS=(
            "--dim_s" "200"
            "--size_m" "32"
        )
    elif [ "$MODEL_NAME" == "dkt+" ]; then
        MODEL_ARGS=(
            "--emb_size" "256"
            "--lambda_r" "1e-3"
            "--lambda_w1" "0.003"
            "--lambda_w2" "3.0"
    )
    else
        echo "Invalid input. Exiting script."
        exit 1
    fi

    MODEL_ARGS=( "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}" )
}

ACTION="eval"

if [ "$ACTION" == "train" ]; then
    MODEL_NAME="dkt+"
    DATASET_NAME="ednet"
    set_model_args
else
    MODEL_ARGS=(
        "--save_dir" "./saved_model/train_dkt+_ednet_qid_saved_model_42_0_0.2_256_0.001_0.001_0.003_3.0/"
        "--test_filename" "test.csv"
        "--use_pred" "0"
        "--train_ratio" "0.9"
        "--atkt_pad" "0" 
    )
fi

if [ "$ACTION" == "train" ]; then
    python3 "$SCRIPT_PATH" "$ACTION" "$MODEL_NAME" "${ARGS[@]}"
else
    python3 "$SCRIPT_PATH" "$ACTION" "${MODEL_ARGS[@]}"
fi
