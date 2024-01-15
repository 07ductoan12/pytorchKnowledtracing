import sys
from pytorchKT import train, eval

if __name__ == "__main__":
    if sys.argv[1] == "train":
        print(f"Start TRAIN {sys.argv[2]} process...\n")
        train()
    elif sys.argv[1] == "eval":
        print("Start EVAL process...\n")
        eval()
    else:
        print("Invalid action. Use 'train' or 'eval'.")
