""" fairseq用にコーパスを変換する """

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from tokenizer import tokenizer


def convert_corpus(corpus_type: str):
    X = []
    y = []

    df = pd.read_csv(f"assets/corpus/tsv/{corpus_type}.tsv", sep="\t")
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    df["source"] = df["source"].map(lambda x: tokenizer(x))
    df["target"] = df["target"].map(lambda x: tokenizer(x))

    for i in range(len(df)):
        srctgt = df.iloc[i, 0:2].values
        src = srctgt[0].replace(" ", "")
        tgt = srctgt[1].replace(" ", "")

        X.append("[SPK1]" + src)
        y.append(tgt)

    X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=0.8)
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other)
    Xy_list = [X_train, X_val, X_test, y_train, y_val, y_test]
    data_categorist = ["src", "dst"]
    split_categorist = ["train", "valid", "test"]

    i = 0
    folderpath = f"assets/corpus/faitseq/{corpus_type}/raw"
    for data in data_categorist:
        for split in split_categorist:
            filepath = os.path.join(folderpath, f"{split}.{data}")
            with open(filepath, mode="w") as f:
                for line in Xy_list[i]:
                    f.write(line + "\n")
            i += 1


def main():
    for corpus_type in ["neg", "pos", "neu"]:
        convert_corpus(corpus_type)


if __name__ == "__main__":
    main()
