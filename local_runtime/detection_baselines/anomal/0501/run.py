import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from common import run_0501, resolve_data_root

DATASETS = {
    "oilrig": ("oilrig_anomal.csv", 3790504),
    "sandworm": ("sandworm_anomal.csv", 3802012),
    "wizard": ("wizard_spider_anomal.csv", 3807085),
    "wizard_spider": ("wizard_spider_anomal.csv", 3807085),
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="oilrig", help="Dataset label")
    ap.add_argument("--file", type=str, default=None, help="Path to 0501 CSV file")
    ap.add_argument("--data-root", type=str, default=None, help="Root dir for anomal data")
    ap.add_argument("--split", type=int, default=None, help="Train/test split index")
    ap.add_argument("--dimension", type=int, default=128, help="DGI output dimension")
    ap.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    ap.add_argument("--epochs", type=int, default=4000, help="Max epochs")
    ap.add_argument("--patience", type=int, default=5, help="Early stop patience")
    return ap.parse_args()

def main():
    args = parse_args()
    dataset_key = args.dataset.lower()

    data_root = resolve_data_root(args.data_root)
    if args.file:
        file_path = args.file
        split = args.split if args.split is not None else 0
    else:
        if dataset_key not in DATASETS:
            raise SystemExit("Unknown dataset: {}".format(args.dataset))
        file_name, default_split = DATASETS[dataset_key]
        file_path = os.path.join(data_root, "0501", file_name)
        split = args.split if args.split is not None else default_split

    if not os.path.exists(file_path):
        raise SystemExit("Missing 0501 data file: {}".format(file_path))

    run_0501(
        file_path=file_path,
        split=split,
        ndim_out=args.dimension,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        dataset_label=dataset_key,
    )

if __name__ == "__main__":
    main()
