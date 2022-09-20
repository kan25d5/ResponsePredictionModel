def __get_corpus_from_twitter(sentiment_type, maxlen, transform):
    from utilities.utility_functions import load_json

    X, y = [], []
    filepath = f"assets/corpus/{sentiment_type}.json"
    corpus = load_json(filepath)

    for msg, res in zip(corpus["X"], corpus["y"]):
        if len(msg) > maxlen or len(res) > maxlen:
            continue
        if len(msg) <= 1 or len(res) <= 1:
            continue
        if transform is not None:
            msg = transform(msg)
            res = transform(res)
        X.append(msg)
        y.append(res)

    return X, y


def __get_corpus_from_base(sentiment_type, maxlen, transform):
    from utilities.utility_functions import load_json

    X, y = [], []

    # persona
    if sentiment_type == "persona" or sentiment_type == "base":
        assert transform is not None, "personaコーパスにはTwitterTrasnformの指定が必要"
        corpus = load_json("assets/corpus/persona.json")
        for msg, res in zip(corpus["X"], corpus["y"]):
            if len(msg) > maxlen or len(res) > maxlen:
                continue
            if len(msg) <= 1 or len(res) <= 1:
                continue
            msg = transform(msg)
            res = transform(res)
            X.append(msg)
            y.append(res)

    # nucc
    if sentiment_type == "nucc" or sentiment_type == "base":
        corpus = load_json("assets/corpus/nucc.json")
        for msg, res in zip(corpus["X"], corpus["y"]):
            if len(msg) > maxlen or len(res) > maxlen:
                continue
            if len(msg) <= 1 or len(res) <= 1:
                continue
            X.append(msg)
            y.append(res)

    assert len(X) == len(y), "発話と応答のリストサイズが一致しない．"
    return X, y


def get_corpus(sentiment_type: str = "neu", maxlen=80, transform=None):
    """
    コーパスをロードして発話／応答のリストを返す．\n
    return X:List[str], y:List[str]
    """
    if sentiment_type == "neu" or sentiment_type == "neg" or sentiment_type == "pos":
        return __get_corpus_from_twitter(sentiment_type, maxlen, transform)
    elif sentiment_type == "persona" or sentiment_type == "nucc" or sentiment_type == "base":
        return __get_corpus_from_base(sentiment_type, maxlen, transform)
    else:
        raise ValueError("sentiment_typeの値が不正")


def get_dataset(X, y, maxlen: int, data_size: float, train_size=0.8, val_size=0.7):
    from sklearn.model_selection import train_test_split
    from dataloader.twitter_dataset import TwitterDataset

    X, y = X[: int(len(X) * data_size)], y[: int(len(y) * data_size)]
    X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, train_size=val_size)

    train_dataset = TwitterDataset(X_train, y_train, maxlen=maxlen)
    val_dataset = TwitterDataset(X_val, y_val, maxlen=maxlen)
    test_dataset = TwitterDataset(X_test, y_test, maxlen=maxlen)

    all_dataset = [train_dataset, val_dataset, test_dataset]
    return all_dataset


def get_dataloader(all_dataset, vocab, maxlen: int, batch_size: int, num_workers: int):
    """
    all_dataset:List[TwitterDataset]を受け取り，all_dataloader:List[DataLoader]を返す．
    all_datasetは，train/val/testの順で作成されたTwitterDataset型のインスタンスを持つリスト．
    - パラメーター\n
    all_dataset : [train_dataset, val_dataset, test_dataset]
    - 戻り値\n
    all_dataloader : [train_dataloader, val_dataloader, test_dataloader]
    """

    from torch.utils.data import DataLoader
    from dataloader.twitter_dataset import collate_fn

    train_dataloader = DataLoader(
        all_dataset[0],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, vocab, maxlen),
    )
    val_dataloader = DataLoader(
        all_dataset[1],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, vocab, maxlen),
    )
    test_dataloader = DataLoader(
        all_dataset[2],
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, vocab, maxlen),
    )
    train_dataloader_callback = DataLoader(
        all_dataset[0],
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, vocab, maxlen),
    )

    all_dataloader = [
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_dataloader_callback,
    ]
    return all_dataloader
