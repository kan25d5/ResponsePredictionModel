CHECK_POINT = "assets/checkpoint/persona.ckpt"


def _init_boilerplate():
    # --------------------------------------
    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    # --------------------------------------
    import os

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def _get_vocab(args):
    from utilities.constant import CHAR2ID_FILEPATH
    from vocab.twitter_vocab import TwitterVocab

    make_vocab = args.make_vocab
    sentiment_type = args.sentiment_type
    maxlen = args.maxlen
    vocab_size = args.vocab_size

    vocab = TwitterVocab()
    if not make_vocab:
        vocab.load_char2id_pkl(CHAR2ID_FILEPATH)
    else:
        from utilities.training_functions import get_corpus

        X, y = get_corpus(sentiment_type=sentiment_type, maxlen=maxlen)
        # data_size = args.data_size
        # X, y = X[: int(len(X) * data_size)], y[: int(len(y) * data_size)]

        vocab.fit(X, y, is_wakati=True)
        vocab.reduce_vocabulary(vocab_size)

    return vocab


def _get_dataloader(args, vocab):
    from utilities.training_functions import get_corpus, get_dataloader, get_dataset

    sentiment_type = args.sentiment_type
    maxlen = args.maxlen
    data_size = args.data_size
    batch_size = args.batch_size
    num_workers = args.num_workers

    # データセットを取得
    transform = None
    if sentiment_type == "persona" or sentiment_type == "base":
        from dataloader.twitter_transform import TwitterTransform

        transform = TwitterTransform()

    X, y = get_corpus(sentiment_type=sentiment_type, maxlen=maxlen, transform=transform)
    all_dataset = get_dataset(X, y, maxlen, data_size)

    # データローダーを取得
    # -> [train, val, test, callback_train]の4つのDataLoaderを取得する
    all_dataloader = get_dataloader(all_dataset, vocab, maxlen, batch_size, num_workers)
    return all_dataloader


def _get_model(args, vocab):
    import torch
    from models.seq2seq_transform import Seq2Seq
    from utilities.utility_functions import load_json

    vocab_size = args.vocab_size
    params = args.params
    maxlen = args.maxlen
    sentiment_type = args.sentiment_type
    beam_size = args.beam_size

    input_dim = vocab_size
    output_dim = vocab_size

    if params == "":
        model = Seq2Seq(input_dim, output_dim, maxlen=maxlen, beam_size=beam_size)
    else:
        param = load_json(params)
        model = Seq2Seq(
            input_dim,
            output_dim,
            maxlen=maxlen,
            beam_size=beam_size,
            pe_dropout=param["pe_dropout"],
            encoder_dropout=param["encoder_dropout"],
            decoder_dropout=param["decoder_dropout"],
            learning_ratio=param["learning_ratio"],
            encoder_num_layers=param["encoder_num_layers"],
            decoder_num_layers=param["decoder_num_layers"],
        )

    if sentiment_type == "neg" or sentiment_type == "pos":
        model.load_state_dict(torch.load(CHECK_POINT)["state_dict"])

    return model


def _get_trainer(args, vocab, dataloader_train_callback, dataloader_test):
    import os

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from utilities.callbacks import DisplaySystenResponses

    sentiment_type = args.sentiment_type
    devices = args.devices
    max_epochs = args.max_epochs
    strategy = args.strategy
    accelerator = args.accelerator

    # コールバックの定義
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True),
        ModelCheckpoint(
            dirpath="assets/checkpoint", filename=sentiment_type, monitor="val_loss", verbose=True
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
        max_epochs=max_epochs,
        strategy=strategy,
        accelerator=accelerator,
    )

    return trainer


def train(args):
    # おまじないコード
    _init_boilerplate()

    # 語彙マップクラスを取得
    vocab = _get_vocab(args)

    # データローダーを取得
    all_dataloader = _get_dataloader(args, vocab)
    dataloader_train = all_dataloader[0]
    dataloader_val = all_dataloader[1]
    dataloader_test = all_dataloader[2]
    dataloader_train_callback = all_dataloader[3]
    dataloader_val_callback = all_dataloader[4]

    # モデルを取得
    model = _get_model(args, vocab)

    # Trainerの定義
    trainer = _get_trainer(args, vocab, dataloader_train_callback, dataloader_val_callback)

    # 学習コード
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(model, dataloader_test)
