import sys
from pytorchKT import train, eval

if __name__ == "__main__":
    if sys.argv[1] == "train":
        print("Start TRAIN process...\n")
        train()
    elif sys.argv[1] == "eval":
        print("Start EVAL process...\n")
        eval()
    else:
        print("Invalid action. Use 'train' or 'eval'.")
