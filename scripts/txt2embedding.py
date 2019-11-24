#!/usr/bin/env python
# coding: utf-8
import pathlib

import h5py
import numpy as np
import pandas as pd
import spacy

DATASET_PATH = pathlib.Path("../data/ag_news_csv")
TRAIN_CSV = DATASET_PATH / "train.csv"
TEST_CSV = DATASET_PATH / "test.csv"


def text2embed(df, model):
    # In the AG news dataset, no title is longer than 20 words
    # Embedding dataset is thus [n_samples, n_words, embedding_dim]
    classes = sorted(df["class"].unique())
    print("Found {} classes: {}".format(len(classes), classes))

    embed = np.zeros([len(df), 20, 300])
    labels = np.zeros([len(df), len(classes)])
    print("Converting {} rows to embeddings...".format(len(df)))
    for ridx, row in df.iterrows():
        if ridx % 5000 == 0:
            print("processed {} / {}...".format(ridx, len(df)))
        title = row["title"]
        label = row["class"]
        for widx, word in enumerate(title.split()):
            vec = model(word).vector.tolist()
            embed[ridx, widx] = vec
        # One-hot encoding of class
        labels[ridx, classes.index(label)] = 1
    return embed, labels


def convert():
    train_h5_file = "train.h5"
    test_h5_file = "test.h5"
    spacy_model = "en_core_web_lg"

    print("Loading spacy model '{}' containing GloVe vectors...".format(spacy_model))
    nlp = spacy.load(spacy_model)
    print("Spacy model loaded.")

    print("Reading train dataset from csv...")
    train = pd.read_csv(TRAIN_CSV, names=["class", "title", "description"])
    print("Reading test dataset from csv...")
    test = pd.read_csv(TEST_CSV, names=["class", "title", "description"])

    print("Converting text in train dataset to arrays of word vector embeddedings...")
    print("-- NOTE: THIS MAY TAKE A LONG TIME.")
    train_embed, train_label = text2embed(train, model=nlp)
    print("Text to embeddings conversion of train dataset is complete.")
    print(
        "Converted train dataset shape: {} (labels: {})".format(
            train_embed.shape, train_label.shape
        )
    )

    print("Converting text in test dataset to arrays of word vector embeddedings...")
    print("-- NOTE: THIS MAY TAKE A LONG TIME.")
    test_embed, test_label = text2embed(test, model=nlp)
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
    print("Beginning conversion...")
    convert()
    print("Exiting.")


if __name__ == "__main__":
    main()
