CHECK_POINT = "assets/neu.ckpt"

sentiment_type: str
maxlen: int
batch_size: int
max_epoch: int
vocab_size: int
strategy: str
accelerator: str
devices: int
num_worker: int
load_vocab: bool
data_size: float


def _init_boilerplate():
    # --------------------------------------
    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    # --------------------------------------
    import os

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def _set_fields(args):
    global sentiment_type
    global maxlen
    global batch_size
    global max_epoch
    global vocab_size
    global strategy
    global devices
    global num_worker
    global accelerator
    global load_vocab
    global data_size

    sentiment_type = args.sentiment_type
    maxlen = args.maxlen
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    vocab_size = args.vocab_size
    strategy = args.strategy
    accelerator = args.accelerator
    devices = args.devices
    num_worker = args.num_worker
    load_vocab = args.load_vocab
    data_size = args.data_size


def _get_vocab():
    from vocab.twitter_vocab import TwitterVocab
    from utilities.constant import CHAR2ID_FILEPATH

    vocab = TwitterVocab()

    if load_vocab:
        vocab.load_char2id_pkl(CHAR2ID_FILEPATH)
    else:
        from utilities.training_functions import get_corpus

        X, y = get_corpus(sentiment_type=sentiment_type, maxlen=maxlen)
        X, y = X[: int(len(X) * data_size)], y[: int(len(y) * data_size)]

        vocab.fit(X, y, is_wakati=True)
        vocab.reduce_vocabulary(vocab_size)

    return vocab


def _get_dataloader(vocab):
    from utilities.training_functions import get_corpus, get_dataset, get_dataloader

    # データセットを取得
    X, y = get_corpus(sentiment_type=sentiment_type, maxlen=maxlen)
    all_dataset = get_dataset(X, y, maxlen, data_size)

    # データローダーを取得
    # -> [train, val, test, callback_train]の4つのDataLoaderを取得する
    all_dataloader = get_dataloader(all_dataset, vocab, maxlen, batch_size, num_worker)
    return all_dataloader


def _get_model(vocab):
    import torch
    from models.seq2seq_transform import Seq2Seq

    input_dim = vocab_size
    output_dim = vocab_size

    model = Seq2Seq(input_dim, output_dim, maxlen=maxlen)
    if sentiment_type == "neg" or sentiment_type == "pos":
        model.load_state_dict(torch.load(CHECK_POINT)["state_dict"])

    return model


def _get_trainer(vocab, dataloader_train_callback, dataloader_test):
    import os
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from utilities.callbacks import DisplaySystenResponses

    # コールバックの定義
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True),
        ModelCheckpoint(
            dirpath="assets", filename=sentiment_type, monitor="val_loss", verbose=True
        ),
        DisplaySystenResponses(vocab, dataloader_train_callback, dataloader_test),
    ]

    # Loggerの定義
    logger = TensorBoardLogger(os.path.join(os.getcwd(), "logs/"), sentiment_type)

    # Trainerの定義
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices=devices,
        max_epochs=max_epoch,
        strategy=strategy,
        accelerator=accelerator,
    )

    return trainer


def train(args):
    # おまじないコード
    _init_boilerplate()

    # コマンドライン引数のフィールド値のセット
    _set_fields(args)

    # 語彙マップクラスを取得
    vocab = _get_vocab()

    # データローダーを取得
    all_dataloader = _get_dataloader(vocab)
    dataloader_train = all_dataloader[0]
    dataloader_val = all_dataloader[1]
    dataloader_test = all_dataloader[2]
    dataloader_train_callback = all_dataloader[3]

    # モデルを取得
    model = _get_model(vocab)

    # Trainerの定義
    trainer = _get_trainer(vocab, dataloader_train_callback, dataloader_test)

    # 学習コード
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(model, dataloader_test)
