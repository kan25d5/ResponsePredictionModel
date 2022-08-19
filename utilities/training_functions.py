from typing import List
from utilities.utility_functions import load_json
from vocab.twitter_vocab import TwitterVocab


def get_corpus(sentiment_type: str = "normal", maxlen=80):
    X, y = [], []
    filepath = f"assets/{sentiment_type}.json"
    corpus = load_json(filepath)

    for msg, res in zip(corpus["X"], corpus["y"]):
        if len(msg) > maxlen or len(res) > maxlen:
            continue
        X.append(msg)
        y.append(res)

    return X, y


def get_dataset(X, y, train_size=0.8, val_size=0.7, maxlen=140, transform=None):
    from sklearn.model_selection import train_test_split
    from dataloader.twitter_dataset import TwitterDataset

    X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, train_size=val_size)

    train_dataset = TwitterDataset(X_train, y_train, transform, maxlen)
    val_dataset = TwitterDataset(X_val, y_val, transform, maxlen)
    test_dataset = TwitterDataset(X_test, y_test, transform, maxlen)
    all_dataset = [train_dataset, val_dataset, test_dataset]

    return all_dataset


def get_dataloader(all_dataset, batch_size=100, num_workers=52, pin_memory=True):
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        all_dataset[0],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=all_dataset[0].collate_fn,
    )
    val_dataloader = DataLoader(
        all_dataset[1],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=all_dataset[1].collate_fn,
    )
    test_dataloader = DataLoader(
        all_dataset[2],
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=all_dataset[2].collate_fn,
    )
    all_dataloader = [train_dataloader, val_dataloader, test_dataloader]
    return all_dataloader


def get_dataloader_pipeline(
    vocab: TwitterVocab,
    sentiment_type: str = "normal",
    maxlen=80,
    train_size=0.8,
    val_size=0.7,
    transform=None,
    batch_size=100,
    num_workers=52,
    verbose=False,
):
    from sklearn.model_selection import train_test_split
    from dataloader.twitter_dataset import TwitterDataset

    # コーパスを取得
    messages, responses = get_corpus(sentiment_type, maxlen)

    # 前処理
    new_msg, new_res = [], []
    if transform is not None:
        for x, y in zip(messages, responses):
            new_msg.append(x)
            new_res.append(y)
        messages = new_msg
        responses = new_res

    # データセットの分割
    X_train, X_train_other, y_train, y_train_other = train_test_split(
        messages, responses, train_size=train_size
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_train_other, y_train_other, train_size=val_size
    )

    # テキストをベクトル化
    X_train = vocab.vocab_X.transform(X_train, is_wakati=False, verbose=verbose)
    y_train = vocab.vocab_X.transform(y_train, is_wakati=False, verbose=verbose)
    X_val = vocab.vocab_X.transform(X_val, is_wakati=False, verbose=verbose)
    y_val = vocab.vocab_X.transform(y_val, is_wakati=False, verbose=verbose)
    y_test = vocab.vocab_X.transform(y_test, is_wakati=False, verbose=verbose)
    X_test = vocab.vocab_X.transform(X_test, is_wakati=False, verbose=verbose)

    # DataSetの作成
    train_dataset = TwitterDataset(X_train, y_train, maxlen)
    val_dataset = TwitterDataset(X_val, y_val, maxlen)
    test_dataset = TwitterDataset(X_test, y_test, maxlen)
    all_dataset = [train_dataset, val_dataset, test_dataset]

    # DataLoaderの作成
    all_dataloader = get_dataloader(all_dataset,batch_size=batch_size, num_workers=num_workers)

    return all_dataloader
