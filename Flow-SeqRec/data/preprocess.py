"""Preprocessing scripts for Amazon-Books and MovieLens-25M datasets."""

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _filter_by_min_interactions(
    interactions: List[Tuple[int, int, float]],
    min_interactions: int = 5,
) -> Tuple[List[Tuple[int, int, float]], Dict[int, int], Dict[int, int]]:
    """Iteratively filter users and items with fewer than min_interactions."""
    df = pd.DataFrame(interactions, columns=["user", "item", "timestamp"])

    changed = True
    while changed:
        before = len(df)
        user_counts = df["user"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df["user"].isin(valid_users)]

        item_counts = df["item"].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[df["item"].isin(valid_items)]
        changed = len(df) < before

    user_map = {u: i + 1 for i, u in enumerate(sorted(df["user"].unique()))}
    item_map = {it: i + 1 for i, it in enumerate(sorted(df["item"].unique()))}

    df["user"] = df["user"].map(user_map)
    df["item"] = df["item"].map(item_map)

    filtered = list(df.itertuples(index=False, name=None))
    return filtered, user_map, item_map


def _build_sequences(
    interactions: List[Tuple[int, int, float]],
) -> Dict[int, List[int]]:
    """Group interactions by user and sort by timestamp."""
    user_items = defaultdict(list)
    for user, item, ts in interactions:
        user_items[user].append((ts, item))

    sequences = {}
    for user, items in user_items.items():
        items.sort(key=lambda x: x[0])
        sequences[user] = [it for _, it in items]
    return sequences


def _split_sequences(
    sequences: Dict[int, List[int]],
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Leave-one-out style split: last item for test, second-to-last for valid.

    For training, we generate all sub-sequences of user history up to
    (but not including) the validation item.
    """
    train_seqs, valid_seqs, test_seqs = [], [], []

    for user, seq in sequences.items():
        if len(seq) < 3:
            continue

        test_seqs.append(seq[:])
        valid_seqs.append(seq[:-1])

        for end_idx in range(2, len(seq) - 1):
            train_seqs.append(seq[: end_idx + 1])

    return train_seqs, valid_seqs, test_seqs


def _save_dataset(
    data_dir: str,
    train_seqs: List[List[int]],
    valid_seqs: List[List[int]],
    test_seqs: List[List[int]],
    num_users: int,
    num_items: int,
    sparse_features: np.ndarray = None,
) -> None:
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "train_sequences.pkl"), "wb") as f:
        pickle.dump(train_seqs, f)
    with open(os.path.join(data_dir, "valid_sequences.pkl"), "wb") as f:
        pickle.dump(valid_seqs, f)
    with open(os.path.join(data_dir, "test_sequences.pkl"), "wb") as f:
        pickle.dump(test_seqs, f)

    meta = {"num_users": num_users, "num_items": num_items}
    with open(os.path.join(data_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(meta, f)

    if sparse_features is not None:
        np.save(os.path.join(data_dir, "item_sparse_features.npy"), sparse_features)

    print(f"Saved preprocessed data to {data_dir}")
    print(f"  num_users={num_users}, num_items={num_items}")
    print(f"  train={len(train_seqs)}, valid={len(valid_seqs)}, test={len(test_seqs)}")


def preprocess_amazon_books(
    raw_path: str,
    output_dir: str = "./data/amazon_books",
    min_interactions: int = 5,
) -> None:
    """Preprocess Amazon-Books dataset.

    Expects a CSV or JSON-lines file with columns: reviewerID, asin, unixReviewTime.
    """
    print(f"Loading Amazon-Books from {raw_path} ...")

    if raw_path.endswith(".csv"):
        df = pd.read_csv(raw_path)
    else:
        df = pd.read_json(raw_path, lines=True)

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "user" in cl or "reviewer" in cl:
            col_map["user"] = c
        elif "item" in cl or "asin" in cl or "product" in cl:
            col_map["item"] = c
        elif "time" in cl or "timestamp" in cl:
            col_map["timestamp"] = c

    assert "user" in col_map and "item" in col_map and "timestamp" in col_map, (
        f"Cannot auto-detect columns. Found: {list(df.columns)}"
    )

    user_ids = {u: i for i, u in enumerate(df[col_map["user"]].unique())}
    item_ids = {it: i for i, it in enumerate(df[col_map["item"]].unique())}

    interactions = []
    for _, row in df.iterrows():
        interactions.append(
            (user_ids[row[col_map["user"]]],
             item_ids[row[col_map["item"]]],
             float(row[col_map["timestamp"]]))
        )

    filtered, user_map, item_map = _filter_by_min_interactions(
        interactions, min_interactions
    )
    sequences = _build_sequences(filtered)
    train, valid, test = _split_sequences(sequences)

    num_users = len(user_map)
    num_items = len(item_map)

    _save_dataset(output_dir, train, valid, test, num_users, num_items)


def preprocess_movielens_25m(
    raw_path: str,
    output_dir: str = "./data/movielens_25m",
    min_interactions: int = 5,
) -> None:
    """Preprocess MovieLens-25M dataset.

    Expects ratings.csv with columns: userId, movieId, rating, timestamp.
    """
    print(f"Loading MovieLens-25M from {raw_path} ...")
    df = pd.read_csv(raw_path)

    interactions = []
    for _, row in df.iterrows():
        interactions.append((int(row["userId"]), int(row["movieId"]), float(row["timestamp"])))

    filtered, user_map, item_map = _filter_by_min_interactions(
        interactions, min_interactions
    )
    sequences = _build_sequences(filtered)
    train, valid, test = _split_sequences(sequences)

    num_users = len(user_map)
    num_items = len(item_map)

    # Build sparse genre features if genome-scores exist
    sparse_features = None
    genome_path = os.path.join(os.path.dirname(raw_path), "genome-scores.csv")
    if os.path.exists(genome_path):
        print("Building sparse item features from genome-scores ...")
        genome_df = pd.read_csv(genome_path)
        tag_ids = sorted(genome_df["tagId"].unique())
        tag_map = {t: i for i, t in enumerate(tag_ids)}
        feat_dim = len(tag_ids)

        features = np.zeros((num_items + 1, feat_dim), dtype=np.float32)
        for _, row in genome_df.iterrows():
            mid = int(row["movieId"])
            if mid in item_map:
                features[item_map[mid], tag_map[int(row["tagId"])]] = float(
                    row["relevance"]
                )
        sparse_features = features

    _save_dataset(output_dir, train, valid, test, num_users, num_items, sparse_features)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess datasets for Flow-SeqRec")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["amazon_books", "movielens_25m"],
    )
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--min_interactions", type=int, default=5)
    args = parser.parse_args()

    if args.dataset == "amazon_books":
        out = args.output_dir or "./data/amazon_books"
        preprocess_amazon_books(args.raw_path, out, args.min_interactions)
    else:
        out = args.output_dir or "./data/movielens_25m"
        preprocess_movielens_25m(args.raw_path, out, args.min_interactions)
