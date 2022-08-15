from typing import List
from utilities.utility_functions import load_json


def get_corpus(sentiment_type: str = "normal"):
    filepath = f"assets/{sentiment_type}.json"
    corpus = load_json(filepath)
    return corpus["X"], corpus["y"]


def get_dataset(X, y, vocab, train_size=0.8, val_size=0.7, maxlen=140, transform=None):
    from sklearn.model_selection import train_test_split
    from dataloader.twitter_dataset import TwitterDataset

    X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_other, y_other, train_size=val_size
    )

    train_dataset = TwitterDataset(X_train, y_train, vocab, transform, maxlen)
    val_dataset = TwitterDataset(X_val, y_val, vocab, transform, maxlen)
    test_dataset = TwitterDataset(X_test, y_test, vocab, transform, maxlen)
    all_dataset = [train_dataset, val_dataset, test_dataset]

    return all_dataset


def get_dataloader(all_dataset, batch_size=100):
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        all_dataset[0],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=all_dataset[0].collate_fn,
    )
    val_dataloader = DataLoader(
        all_dataset[1],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=all_dataset[1].collate_fn,
    )
    test_dataloader = DataLoader(
        all_dataset[2],
        batch_size=1,
        shuffle=False,
        collate_fn=all_dataset[2].collate_fn,
    )
    all_dataloader = [train_dataloader, val_dataloader, test_dataloader]
    return all_dataloader

