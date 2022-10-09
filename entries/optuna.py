import os
from typing import List

import pytorch_lightning as pl
from dataloader.twitter_dataset import TwitterDataset
from dataloader.twitter_transform import TwitterTransform
from torch.utils.data import DataLoader
from utilities.utility_functions import load_json, save_json
from vocab.twitter_vocab import TwitterVocab

import optuna

OPTUNA_MAX_EPOCH = 20

messages: List[str]
responses: List[str]
all_dataset: List[TwitterDataset]
all_dataloader: List[DataLoader]

vocab: TwitterVocab
transform = TwitterTransform()


def _get_corpus(sentiment_type, maxlen, transform=None):
    messages = []
    responses = []
    corpus = load_json(f"assets/corpus/{sentiment_type}.json")

    for msg, res in zip(corpus["X"], corpus["y"]):
        if transform is not None:
            msg = transform(msg)
            res = transform(res)
        if len(msg) <= 1 or len(res) <= 1:
            continue
        if len(msg) > maxlen or len(res) > maxlen:
            continue
        messages.append(msg)
        responses.append(res)

    return messages, responses


def _get_corpus_persona(maxlen, transform=None):
    import pandas as pd
    from utilities.constant import PERSONA_DATASET

    messages = []
    responses = []
    df = pd.read_excel(PERSONA_DATASET)

    persona_ids = df["ペルソナID"].unique()
    for persona_id in persona_ids:
        X, y = [], []

        df_dialoge = df[df["ペルソナID"] == persona_id]
        dialogues = df_dialoge["発話"].values
        for i in range(0, len(dialogues), 2):
            if len(dialogues) == i + 1:
                break
            X.append(dialogues[i])
            y.append(dialogues[i + 1])

        assert len(X) == len(y), "発話リストと応答リストが一致しない．"
        messages.extend(X)
        responses.extend(y)

    print("データサイズ：")
    print("\tmessages : {}".format(len(messages)))
    print("\tresponses : {}".format(len(responses)))

    return messages, responses


def _set_vocabs(args):
    from utilities.training_functions import get_corpus

    global vocab

    maxlen = args.maxlen
    vocab_size = args.vocab_size
    sentiment_type = args.sentiment_type

    transform = TwitterTransform(is_wakati=True)
    vocab = TwitterVocab(max_vocab=vocab_size)

    if args.make_vocab:
        snt_msgs, snt_res = get_corpus(sentiment_type, maxlen)
        normal_X, normal_y = get_corpus("normal", maxlen, transform)

        vocab.fit(snt_msgs, snt_res, verbose=True, is_wakati=False)
        vocab.fit(normal_X, normal_y, verbose=True)

        vocab.save_char2id_pkl(f"assets/char2id/{args.sentiment_type}_char2id.model")
    else:
        vocab.load_char2id_pkl(f"assets/char2id/{args.sentiment_type}_char2id.model")


def _set_dataloader(args):
    global messages
    global responses
    global all_dataset
    global all_dataloader

    from utilities.training_functions import get_dataloader, get_dataset

    maxlen = args.maxlen
    data_size = args.data_size
    batch_size = args.batch_size
    num_workers = args.num_workers

    messages, responses = _get_corpus_persona(maxlen, transform)
    all_dataset = get_dataset(messages, responses, maxlen=maxlen, data_size=data_size)
    all_dataloader = get_dataloader(all_dataset, vocab, maxlen, batch_size, num_workers)


def objective(trial: optuna.Trial, args):
    from models.seq2seq_transform import Seq2Seq
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from utilities.callbacks import DisplaySystenResponses

    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

    pe_dropout = trial.suggest_float("pe_dropout", 0.1, 0.6)
    encoder_dropout = trial.suggest_float("encoder_dropout", 0.1, 0.6)
    decoder_dropout = trial.suggest_float("decoder_dropout", 0.1, 0.6)
    learning_ratio = trial.suggest_float("learning_ratio", 0.00001, 0.001)
    encoder_num_layers = trial.suggest_int("encoder_num_layers", 3, 7)
    decoder_num_layers = trial.suggest_int("decoder_num_layers", 3, 7)

    print("-" * 20)
    print("current params is :")
    for k, v in trial.params.items():
        print("\t{}:{:.6f}".format(k, v))
    print("-" * 20)

    maxlen = args.maxlen
    beam_size = args.beam_size
    input_dim = len(vocab.vocab_X.char2id)
    output_dim = len(vocab.vocab_y.char2id)

    model = Seq2Seq(
        input_dim,
        output_dim,
        maxlen=maxlen,
        beam_size=beam_size,
        pe_dropout=pe_dropout,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        encoder_num_layers=encoder_num_layers,
        decoder_num_layers=decoder_num_layers,
        learning_ratio=learning_ratio,
    )

    train_dataloader = all_dataloader[0]
    val_dataloader = all_dataloader[1]
    test_dataloader = all_dataloader[2]
    train_callback_dataloader = all_dataloader[3]
    val_callback_dataloader = all_dataloader[4]

    logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="optuna",
        version=f"{trial.number}_trial",
    )

    callbacks = [
        EarlyStopping("val_loss", patience=args.patience, verbose=True, mode="min"),
        ModelCheckpoint(
            dirpath="assets/checkpoint/",
            filename=f"{args.sentiment_type}_trial_{trial.number}.ckpt",
        ),
        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        DisplaySystenResponses(
            vocab,
            train_callback_dataloader,
            val_callback_dataloader,
            filename="optuna",
            trial=trial,
        ),
    ]

    # TODO : 2gpusで実行できない．
    #        TypeError: cannot pickle 'Tagger' object が発生．
    #        MeCab.Taggerのことだと思うので，
    #        TwitterTransform.Taggerがよくない？
    # TODO : 最大エポック数を再考
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator=args.accelerator,
        # strategy=args.strategy,
        # devices=args.devices,
        devices=1,
        max_epochs=OPTUNA_MAX_EPOCH,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return trainer.callback_metrics["val_loss"].item()


def run(args):
    _set_vocabs(args)
    _set_dataloader(args)

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    trial = study.best_trial
    print("Params: ")
    for key, value in trial.params.items():
        print("\t{}: {}".format(key, value))

    params = trial.params
    params["best_value"] = trial.value
    save_json(params, "assets/best_models/best_params.json")
