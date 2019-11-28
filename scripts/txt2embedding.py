#!/usr/bin/env python
# coding: utf-8
import argparse
import pathlib

import h5py
import numpy as np
import pandas as pd
import spacy


def text2embed(df, model, max_len, embed_dim):
    # Embedding dataset is [n_samples, n_words, embedding_dim]
    classes = sorted(df["class"].unique())
    print("Found {} classes: {}".format(len(classes), classes))

    embed = np.zeros([len(df), max_len, embed_dim])
    labels = np.zeros([len(df), len(classes)])
    print("Converting {} rows to embeddings...".format(len(df)))
    for ridx, row in df.iterrows():
        if ridx % 5000 == 0:
            print("processed {} / {}...".format(ridx, len(df)))
        text = row["content"]
        label = row["class"]
        for widx, word in enumerate(text.split()):
            if widx >= max_len:
                break
            vec = model(word).vector.tolist()
            embed[ridx, widx] = vec
        # One-hot encoding of class
        labels[ridx, classes.index(label)] = 1
    return embed, labels


def convert(args):
    train_csv = pathlib.Path(args.dataset_path) / "train.csv"
    test_csv = pathlib.Path(args.dataset_path) / "test.csv"

    print("Converting dataset in '{}'".format(str(args.dataset_path)))

    train_h5_file = "train.h5"
    test_h5_file = "test.h5"
    spacy_model = "en_core_web_lg"

    print("Loading spacy model '{}' containing GloVe vectors...".format(spacy_model))
    nlp = spacy.load(spacy_model)
    print("Spacy model loaded.")

    print("Reading train dataset from csv...")
    train = pd.read_csv(train_csv, names=["class", "title", "content"])
    print("Reading test dataset from csv...")
    test = pd.read_csv(test_csv, names=["class", "title", "content"])

    if args.max_samples is None:
        args.max_samples = max([len(train), len(test)])

    if args.shuffle:
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)

    print("Converting text in train dataset to arrays of word vector embeddings...")
    print("-- NOTE: THIS MAY TAKE A LONG TIME.")
    train_embed, train_label = text2embed(
        train[: args.max_samples],
        model=nlp,
        max_len=args.max_len,
        embed_dim=args.embedding_dim,
    )
    print("Text to embeddings conversion of train dataset is complete.")
    print(
        "Converted train dataset shape: {} (labels: {})".format(
            train_embed.shape, train_label.shape
        )
    )

    print("Converting text in test dataset to arrays of word vector embeddedings...")
    print("-- NOTE: THIS MAY TAKE A LONG TIME.")
    test_embed, test_label = text2embed(
        test[: args.max_samples],
        model=nlp,
        max_len=args.max_len,
        embed_dim=args.embedding_dim,
    )
    print("Text to embeddings conversion of test dataset is complete.")
    print(
        "Converted test dataset shape: {} (labels: {})".format(
            test_embed.shape, test_label.shape
        )
    )

    # Save train.h5
    print("Saving train dataset to '{}'...".format(train_h5_file))
    h5f = h5py.File(train_h5_file, "w")
    h5f.create_dataset("train_x", data=train_embed)
    h5f.create_dataset("train_y", data=train_label)
    h5f.close()
    print("Saving train dataset complete.")

    # Save test.h5
    print("Saving test dataset to '{}'...".format(test_h5_file))
    h5f = h5py.File(test_h5_file, "w")
    h5f.create_dataset("test_x", data=test_embed)
    h5f.create_dataset("test_y", data=test_label)
    h5f.close()
    print("Saving test dataset complete.")
    print()
    print("Conversion completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--dataset_path", required=True, help="Path to csv dataset")
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="Maximum word length for documents (everything after is truncated)",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=300, help="Embedding dimensionality"
    )
    parser.add_argument(
        "--max_samples", type=int, help="Maximum number of samples in dataset"
    )
    args = parser.parse_args()

    print("Beginning conversion...")
    convert(args)
    print("Exiting.")


if __name__ == "__main__":
    main()
