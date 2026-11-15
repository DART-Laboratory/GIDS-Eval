import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from common import run_cic, resolve_data_root

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None, help="Path to CIC CSV file")
    ap.add_argument("--data-root", type=str, default=None, help="Root dir for anomal data")
    ap.add_argument("--split", type=int, default=559918, help="Train/test split index")
    ap.add_argument("--dimension", type=int, default=128, help="DGI output dimension")
    ap.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    ap.add_argument("--epochs", type=int, default=4000, help="Max epochs")
    ap.add_argument("--patience", type=int, default=5, help="Early stop patience")
    ap.add_argument("--dataset", type=str, default="cic", help="Dataset label")
    return ap.parse_args()

def main():
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    file_path = args.file or os.path.join(data_root, "cic", "cic_2017_1.csv")
    if not os.path.exists(file_path):
        raise SystemExit("Missing CIC data file: {}".format(file_path))

    run_cic(
        file_path=file_path,
        split=args.split,
        ndim_out=args.dimension,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        dataset_label=args.dataset.lower(),
    )

if __name__ == "__main__":
    main()
