# --------------------------------------
# デフォルト値の設定
# --------------------------------------
MAXLEN = 80
BATCH_SIZE = 80
EPOCH_SIZE = 200
VOCAB_SIZE = 40000
N_TRIALS = 100

STRATEGY = "ddp"
ACCELERATOR = "gpu"
DEVICES = 2
NUM_WORKER = 26
PIN_MEMORY = False

LOAD_MODEL = "assets/ST:normalB:80_E:200_ML:80_VS:40000_base.pth"


# --------------------------------------
# ArgumentParserの設定
# --------------------------------------
import argparse

# パーサーの作成
description = "積極的／消極的な応答を生成するモデル"
parser = argparse.ArgumentParser(description=description)

# ヘルプの定義
help_mode = "起動するモードを選択する．\
    train : 指定したパラメータでモデルの訓練する．\
    pred : 学習済みのモデルを利用してコマンドライン\
    optuna : ハイパーパラメータを探索する．"
help_sentimet = "応答の極性を選択する．\
    pos : 積極的な応答を訓練/生成する．\
    neg : 消極的な応答を訓練/生成する．\
    neural : ニュートラルな応答を訓練/生成する．\
    normal : 全ての応答で訓練する．"
help_maxlen = "応答する系列の最大サイズ．default={}".format(MAXLEN)
help_batch_size = "バッチサイズ．default={}".format(BATCH_SIZE)
help_max_epoch = "最大エポックサイズ．default={}".format(EPOCH_SIZE)
help_vocab_size = "語彙サイズ．default={}".format(VOCAB_SIZE)

# コマンドライン引数の追加
parser.add_argument("mode", help=help_mode, type=str)
parser.add_argument("-st", "--sentiment_type", help=help_sentimet, type=str, default="normal")
parser.add_argument("-len", "--maxlen", help=help_maxlen, type=int, default=MAXLEN)
parser.add_argument("-bt", "--batch_size", help=help_batch_size, type=int, default=BATCH_SIZE)
parser.add_argument("-ep", "--max_epoch", help=help_max_epoch, type=int, default=EPOCH_SIZE)
parser.add_argument("-vs", "--vocab_size", help=help_vocab_size, type=int, default=VOCAB_SIZE)
parser.add_argument("--strategy", type=str, default=STRATEGY)
parser.add_argument("--accelerator", type=str, default=ACCELERATOR)
parser.add_argument("--devices", type=int, default=DEVICES)
parser.add_argument("--num_worker", type=int, default=NUM_WORKER)
parser.add_argument("--n_trials", type=int, default=N_TRIALS)


def train(args):
    # --------------------------------------
    # コマンドライン引数
    # --------------------------------------
    sentiment_type = args.sentiment_type
    maxlen = args.maxlen
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    vocab_size = args.vocab_size
    strategy = args.strategy
    accelerator = args.accelerator
    devices = args.devices
    num_worker = args.num_worker

    # --------------------------------------
    # おまじない
    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    # --------------------------------------
    import os

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # --------------------------------------
    # Vocab / DataLoaderの作成
    # --------------------------------------
    print("Vocab / DataLoaderの作成 : ")
    from vocab.twitter_vocab import TwitterVocab
    from utilities.training_functions import get_dataloader_pipeline

    vocab = TwitterVocab()
    vocab.load_char2id_pkl()
    # char2id.modelは80000語彙の辞書データを持つため，
    # 語彙削減する．
    vocab.reduce_vocabulary(vocab_size)

    all_dataloader = get_dataloader_pipeline(
        vocab,
        sentiment_type=sentiment_type,
        maxlen=maxlen,
        batch_size=batch_size,
        num_workers=num_worker,
        verbose=True,
        pin_memory=PIN_MEMORY,
        is_saved=True,
    )
    train_dataloader = all_dataloader[0]
    val_dataloader = all_dataloader[1]
    test_dataloader = all_dataloader[2]

    # --------------------------------------
    # Modelの作成
    # --------------------------------------
    from models.seq2seq_transform import Seq2Seq

    input_dim = len(vocab.vocab_X.char2id)
    output_dim = len(vocab.vocab_y.char2id)

    model = Seq2Seq(input_dim, output_dim, maxlen=maxlen)

    # --------------------------------------
    # 出力ファイル名の定義
    # --------------------------------------
    filename = "ST:{}B:{}_E:{}_ML:{}_VS:{}".format(
        sentiment_type, batch_size, max_epoch, maxlen, vocab_size
    )

    # --------------------------------------
    # コールバック／ロガーの定義
    # --------------------------------------
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True),
    ]
    logger = TensorBoardLogger(os.path.join(os.getcwd(), "logs/"), filename)

    # --------------------------------------
    # Modelの適合
    # --------------------------------------
    import torch
    import pytorch_lightning as pl
    from multiprocessing import freeze_support

    freeze_support()
    trainer = pl.Trainer(
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        max_epochs=max_epoch,
        logger=logger,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)

    torch.save(model.state_dict(), f"assets/{filename}_base.pth")


def predict(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------
    # コマンドライン引数
    # --------------------------------------
    sentiment_type = args.sentiment_type
    maxlen = args.maxlen
    vocab_size = args.vocab_size
    strategy = args.strategy
    accelerator = args.accelerator
    devices = args.devices
    num_worker = args.num_worker

    # --------------------------------------
    # おまじない
    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    # --------------------------------------
    import os

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # --------------------------------------
    # Vocab / DataLoaderの作成
    # --------------------------------------
    print("Vocab / DataLoaderの作成 : ")
    from vocab.twitter_vocab import TwitterVocab
    from utilities.training_functions import get_dataloader_pipeline

    vocab = TwitterVocab()
    vocab.load_char2id_pkl()
    # char2id.modelは80000語彙の辞書データを持つため，
    # 語彙削減する．
    vocab.reduce_vocabulary(vocab_size)

    # --------------------------------------
    # Modelの作成
    # --------------------------------------
    from models.seq2seq_transform import Seq2Seq
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    input_dim = len(vocab.vocab_X.char2id)
    output_dim = len(vocab.vocab_y.char2id)

    model = Seq2Seq(input_dim, output_dim, maxlen=maxlen).to(device)
    model.load_state_dict(torch.load(LOAD_MODEL))

    # --------------------------------------
    # 推論
    # --------------------------------------
    with torch.no_grad():
        while True:
            input_line = input(":")
            if input_line == "" or input_line == "exit":
                break

            X = vocab.vocab_X.encode(input_line, is_wakati=False)
            X = [[item for item in X]]
            X = pad_sequences(X, maxlen=maxlen, padding="post")
            X = torch.LongTensor(X).t().to(device)

            pred = model(X)
            pred = [item[0] for item in pred.tolist()]
            pred = vocab.vocab_y.decode(pred)
            print("pred : {}".format(pred))


def optuna(args):
    import optuna
    from vocab.twitter_vocab import TwitterVocab

    def objective(trial: optuna.Trial, vocab: TwitterVocab, best_valloss, all_dataloader, args):
        train_dataloader = all_dataloader[0]
        val_dataloader = all_dataloader[1]
        test_dataloader = all_dataloader[2]

        # --------------------------------------
        # コマンドライン引数
        # --------------------------------------
        max_epoch = 20
        maxlen = 80
        strategy = args.strategy
        accelerator = args.accelerator
        devices = args.devices

        # --------------------------------------
        # ハイパラ探索
        # --------------------------------------
        pe_dropout = trial.suggest_float("pe_dropout", 0.1, 0.6)
        encoder_dropout = trial.suggest_float("encoder_dropout", 0.1, 0.6)
        decoder_dropout = trial.suggest_float("decoder_dropout", 0.1, 0.6)
        encoder_num_layers = trial.suggest_int("encoder_num_layers", 3, 9)
        decoder_num_layers = trial.suggest_int("decoder_num_layers", 3, 9)
        learning_ratio = trial.suggest_float("learning_ratio", 1e-7, 1e-4)

        # --------------------------------------
        # モデルの定義
        # --------------------------------------
        from models.seq2seq_transform import Seq2Seq

        input_dim = len(vocab.vocab_X.char2id)
        output_dim = len(vocab.vocab_y.char2id)
        model = Seq2Seq(
            input_dim,
            output_dim,
            maxlen=maxlen,
            pe_dropout=pe_dropout,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
            learning_ratio=learning_ratio,
        )

        # --------------------------------------
        # コールバック／ロガーの定義
        # --------------------------------------
        from pytorch_lightning.callbacks import EarlyStopping
        from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True),
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ]

        # --------------------------------------
        # Modelの適合
        # --------------------------------------
        import torch
        import pytorch_lightning as pl
        from multiprocessing import freeze_support
        from utilities.utility_functions import save_json

        freeze_support()
        trainer = pl.Trainer(
            strategy=strategy,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            max_epochs=max_epoch,
        )
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.test(model, test_dataloader)

        val_loss = trainer.callback_metrics["val_loss"].item()
        if val_loss < best_valloss:
            torch.save(model.state_dict(), "assets/model.pth")
            save_json(trial.params.items(), "assets/best_params.json")

        return val_loss

    # --------------------------------------
    # おまじない
    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    # --------------------------------------
    import os

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # --------------------------------------
    # コマンドライン引数
    # --------------------------------------
    sentiment_type = "neu"
    vocab_size = 8000
    maxlen = 80
    batch_size = args.batch_size
    num_worker = args.num_worker
    n_trials = args.n_trials

    # --------------------------------------
    # Vocab / DataLoaderの作成
    # --------------------------------------
    print("Vocab / DataLoaderの作成 : ")
    from vocab.twitter_vocab import TwitterVocab
    from utilities.training_functions import get_dataloader_pipeline

    vocab = TwitterVocab()
    vocab.load_char2id_pkl()
    # char2id.modelは80000語彙の辞書データを持つため，
    # 語彙削減する．
    vocab.reduce_vocabulary(vocab_size)

    all_dataloader = get_dataloader_pipeline(
        vocab,
        sentiment_type=sentiment_type,
        maxlen=maxlen,
        batch_size=batch_size,
        num_workers=num_worker,
        verbose=True,
        pin_memory=PIN_MEMORY,
        is_saved=True,
    )

    # --------------------------------------
    # trial定義
    # --------------------------------------
    best_valloss = 9999.0
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, vocab, best_valloss, all_dataloader, args),
        n_trials=n_trials,
    )
    trial = study.best_trial

    print("Params : ")
    for key, value in trial.params.items():
        print("\t{}:{}".format(key, value))


def main():
    args = parser.parse_args()
    run_mode = args.mode

    if run_mode == "train":
        train(args)
    elif run_mode == "pred":
        predict(args)
    elif run_mode == "optuna":
        optuna(args)
    else:
        raise ValueError("modeの引数が不正．--helpを参照．")


if __name__ == "__main__":
    main()
