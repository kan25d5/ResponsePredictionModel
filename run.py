# --------------------------------------
# デフォルト値の設定
# --------------------------------------
MAXLEN = 80
BATCH_SIZE = 20
EPOCH_SIZE = 50
VOCAB_SIZE = 8000

STRATEGY = "ddp"
ACCELERATOR = "gpu"
DEVICES = 2
NUM_WORKER = 52 // 2


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


def train(args):
    # --------------------------------------
    # コマンドライン引数
    # --------------------------------------
    sentiment_type = args.sentiment_type
    maxlen = args.maxlen
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    vocab_size = args.vocab_size

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
    vocab.reduce_vocabulary(vocab_size)

    all_dataloader = get_dataloader_pipeline(
        vocab,
        sentiment_type=sentiment_type,
        maxlen=maxlen,
        batch_size=batch_size,
        verbose=True,
        num_workers=NUM_WORKER,
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

    model = Seq2Seq(input_dim, output_dim, vocab, maxlen=maxlen + 8)

    # --------------------------------------
    # コールバックの定義
    # --------------------------------------
    from pytorch_lightning.callbacks import EarlyStopping

    callbacks = [EarlyStopping(monitor="val_loss")]

    # --------------------------------------
    # Modelの適合
    # --------------------------------------
    import torch
    import pytorch_lightning as pl
    from multiprocessing import freeze_support

    freeze_support()
    trainer = pl.Trainer(
        strategy=STRATEGY,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        callbacks=callbacks,
        max_epochs=max_epoch,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)


def main():
    args = parser.parse_args()
    run_mode = args.mode

    if run_mode == "train":
        train(args)
    elif run_mode == "pred":
        # TODO: 対話モードによるモデルの利用
        pass
    elif run_mode == "optuna":
        # TODO: ハイパーパラメータ探索を行う
        pass
    else:
        raise ValueError("modeの引数が不正．--helpを参照．")


if __name__ == "__main__":
    main()
