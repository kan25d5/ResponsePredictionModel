from typing import List
from utilities.utility_functions import load_json
from vocab.twitter_vocab import TwitterVocab


def get_corpus(sentiment_type: str = "normal", maxlen=80, transform=None):
    """ 
    コーパスをロードして発話／応答のリストを返す．\n
    return X:List[str], y:List[str]
    """

    X, y = [], []
    filepath = f"assets/{sentiment_type}.json"
    corpus = load_json(filepath)

    for msg, res in zip(corpus["X"], corpus["y"]):
        if len(msg) > maxlen or len(res) > maxlen:
            continue
        if transform is not None:
            msg = transform(msg)
            res = transform(res)
        X.append(msg)
        y.append(res)

    return X, y


def get_dataloader(
    all_dataset, vocab, maxlen: int, batch_size=100, num_workers=52, pin_memory=True
):
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
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_fn(batch, vocab, maxlen),
    )
    val_dataloader = DataLoader(
        all_dataset[1],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_fn(batch, vocab, maxlen),
    )
    test_dataloader = DataLoader(
        all_dataset[2],
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=pin_memory,
        collate_fn=lambda batch: collate_fn(batch, vocab, maxlen),
    )
    all_dataloader = [train_dataloader, val_dataloader, test_dataloader]
    return all_dataloader


def __get_dataloader_pipline_no_load(
    vocab: TwitterVocab,
    sentiment_type: str = "normal",
    maxlen=80,
    train_size=0.8,
    val_size=0.7,
    transform=None,
    verbose=False,
    is_saved=False,
):
    from sklearn.model_selection import train_test_split
    from dataloader.twitter_dataset import TwitterDataset

    # --------------------------------------
    # 発話／応答リストを取得
    # --------------------------------------
    messages, responses = get_corpus(sentiment_type, maxlen, transform)

    # --------------------------------------
    # データセットの分割
    # --------------------------------------
    X_train, X_train_other, y_train, y_train_other = train_test_split(
        messages, responses, train_size=train_size
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_train_other, y_train_other, train_size=val_size
    )

    # --------------------------------------
    # データの中身を確認
    # --------------------------------------
    def display_Xy(X, y, data_label: str, count=4):
        print(f"check {data_label} data :")
        for idx, (msg, res) in enumerate(zip(X, y)):
            print(msg)
            print(f"\t->{res}")
            if idx >= count:
                break

    if verbose:
        display_Xy(X_train, y_train, "train")
        display_Xy(X_val, y_val, "val")
        display_Xy(X_test, y_test, "test")

    # --------------------------------------
    # DataSetの作成
    # --------------------------------------
    train_dataset = TwitterDataset(X_train, y_train, maxlen)
    val_dataset = TwitterDataset(X_val, y_val, maxlen)
    test_dataset = TwitterDataset(X_test, y_test, maxlen)
    all_dataset = [train_dataset, val_dataset, test_dataset]

    # --------------------------------------
    # datasetをロードする
    # --------------------------------------
    if is_saved:
        train_dataset.save_dataset_pkl("train_dataset.pkl")
        val_dataset.save_dataset_pkl("val_dataset.pkl")
        test_dataset.save_dataset_pkl("test_dataset.pkl")

    return all_dataset


def __get_dataloader_pipline_load():
    from dataloader.twitter_dataset import TwitterDataset

    data_labels = ["train", "val", "test"]
    all_datasets = []

    for label in data_labels:
        dataset = TwitterDataset()
        dataset.load_dataset_pkl(f"{label}_dataset.pkl")
        all_datasets.append(dataset)

    return all_datasets


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
    pin_memory=True,
    is_saved=False,
    is_load=False,
):
    """dataloaderを一括で受け取る．
    
    Parameter
    ----------------------
    - vocab : TwitterVocab 辞書データロード済みの語彙マップクラス
    - sentiment_type: str 利用するコーパスタイプ
    - maxlen: int 最大系列数（文字数）．maxlen+1以上の文字数を持つ対話ターンは削減する．
    - train_size: float 
    - val_size: float
    - transform: 前処理クラス
    - batch_size: int バッチサイズ
    - num_workers
    - verbose
    - is_saved: bool ロードしたデータセットのpklデータをセーブする
    - is_load: bool データセットのpklをロードする．
    """

    if is_load:
        all_dataset = __get_dataloader_pipline_load()
    else:
        all_dataset = __get_dataloader_pipline_no_load(
            vocab, sentiment_type, maxlen, train_size, val_size, transform, verbose, is_saved
        )

    # DataLoaderの作成
    all_dataloader = get_dataloader(
        all_dataset, vocab, maxlen, batch_size, num_workers, pin_memory
    )

    return all_dataloader
