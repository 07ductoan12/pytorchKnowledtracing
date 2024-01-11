#!/bin/bash

SCRIPT_PATH=./main.py

ACTION="eval"

# Set the arguments for your script
if [ "$ACTION" == "train" ]; then
    ARGS=( "--dataset_name" "ednet"
           "--model_name" "dkvmn"
           "--emb_type" "qid"
           "--save_dir" "saved_model"
           "--seed" "42"
           "--fold" "0"
           "--dropout" "0.2"
           "--emb_size" "256"
           "--learning_rate" "1e-3" 
           "--dim_s" "128"
           "--size_m" "32"
           "--num_en" "1"
           "--difficult_levels" "100"
           "--num_steps" "199"
           )
else
    ARGS=( "--save_dir" "./saved_model/train_ednet_dkvmn_qid_saved_model_42_0_0.2_128_256_0.001_8_1_32_64_199_100"
           "--test_filename" "test.csv"
           "--use_pred" "0"
           "--train_ratio" "0.9"
           "--atkt_pad" "0" )
fi

# Run the Python script with the specified arguments
python3 $SCRIPT_PATH $ACTION "${ARGS[@]}"