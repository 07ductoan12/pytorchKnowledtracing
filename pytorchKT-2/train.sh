SCRIPT_PATH=./main.py

# Set the arguments for your script
ARGS="--dataset_name assist2015
      --model_name sakt
      --emb_type qid
      --save_dir saved_model
      --seed 42
      --fold 0
      --dropout 0.2
      --emb_size 256
      --learning_rate 1e-3
      "

# Run the Python script with the specified arguments
python3 $SCRIPT_PATH $ARGS