import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchtext.vocab import Vocab

from models.seq2seq_transformer import Seq2SeqTransformer
from utilities.callbacks import DisplayPredictedResponse
from utilities.training_functions import (
    get_corpus_df,
    get_dataloader,
    get_datasets,
    get_transform,
    load_vocabs,
)

CHECK_POINT = "assets/checkpoint/persona.ckpt"

# フィールド変数
source_vocab: Vocab
target_vocab: Vocab


def _init_boilerplate(args):
    # --------------------------------------
    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    # --------------------------------------
    import os

    os.environ["MKL_NUM_THREADS"] = "0"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # ------------------------------------
    # Reference:
    # https://github.com/Lightning-AI/lightning/issues/1314#issuecomment-706607614
    # ------------------------------------
    device_count = args.devices
    if device_count <= 0:
        raise ValueError("デバイスの指定が0以下です．--device={}".format(device_count))
    elif device_count == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(device_count)])


def check_dataloader(dataloader):
    for idx, batch in enumerate(dataloader):
        if idx > 5:
            break

        x, t = batch

        x_item = x[:, 0].tolist()
        t_item = t[:, 0].tolist()

        source = "".join(
            [token for token in source_vocab.lookup_tokens(x_item) if token != "<pad>"]
        )
        target = "".join(
            [token for token in target_vocab.lookup_tokens(t_item) if token != "<pad>"]
        )
        print("source : {}".format(source))
        print("target : {}".format(target))
        print("-" * 20)


def training_data_pipeline(args):
    global source_vocab
    global target_vocab

    # コーパスをロード
    df = get_corpus_df(args.sentiment_type)

    # Source, Targetの
    # 語彙セットクラスtorchtext.vocab.Vocabを作成
    # source_vocab, target_vocabはフィールド変数
    source_vocab, target_vocab = load_vocabs()

    # torchtext.Transformを作成
    source_transform, target_transform = get_transform(source_vocab, target_vocab)

    # コーパスを訓練/検証/テストの３つに分割し
    # DataLoaderが読み込める形にする
    all_dataset = get_datasets(df)

    # データローダーを作成する
    all_dataloader = get_dataloader(
        all_dataset, source_transform, target_transform, args.batch_size
    )

    # データローダーの中身を確認する
    train_dataloader = all_dataloader[0]
    val_dataloader = all_dataloader[1]
    test_dataloader = all_dataloader[2]

    check_dataloader(train_dataloader)
    check_dataloader(val_dataloader)
    check_dataloader(test_dataloader)

    return all_dataloader


def get_model(args):
    src_vocab_size = len(source_vocab.get_stoi())
    tgt_vocab_size = len(target_vocab.get_stoi())

    print("src_vocab_size : {}".format(src_vocab_size))
    print("tgt_vocab_size : {}".format(tgt_vocab_size))

    model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size, beam_size=args.beam_size)
    if args.sentiment_type != "persona":
        # personaモデルのトレーニング以外はファインチューニングする
        model.load_state_dict(torch.load(CHECK_POINT)["state_dict"])
    return model


def get_trainer(args, test_callback_dataloader):
    logger = TensorBoardLogger(os.path.join(os.getcwd(), "logs/"), args.sentiment_type)
    callbacks = [
        ModelCheckpoint(
            dirpath="assets/checkpoint", filename=args.sentiment_type, monitor="val_loss"
        ),
        EarlyStopping(monitor="val_loss", patience=args.patience),
        DisplayPredictedResponse(source_vocab, target_vocab, test_callback_dataloader),
    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices=args.devices,
        max_epochs=args.max_epoch,
        accelerator=args.accelerator,
        strategy=args.strategy if args.strategy != "" or args.strategy != "None" else None,
    )

    return trainer


def train(args):
    # GPU check
    assert torch.cuda.is_available(), "GPUが認識されていない"

    # おまじないコード
    _init_boilerplate(args)

    # データローダーを取得する
    all_dataloader = training_data_pipeline(args)

    train_dataloader = all_dataloader[0]
    val_dataloader = all_dataloader[1]
    test_dataloader = all_dataloader[2]
    test_callback_dataloader = all_dataloader[3]

    # モデルをロード
    model = get_model(args)

    # pl.Trainerを取得
    trainer = get_trainer(args, test_callback_dataloader)

    # モデルをトレーニング
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)
