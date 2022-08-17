# --------------------------------------
# 定数
# --------------------------------------
MAXLEN = 80
BATCH_SIZE = 50
EPOCH_SIZE = 50
VOCAB_SIZE = 10000


def train():
    import os

    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # --------------------------------------
    # Vocab / DataLoaderの作成
    # --------------------------------------
    from vocab.twitter_vocab import TwitterVocab
    from utilities.training_functions import get_dataloader_pipeline

    vocab = TwitterVocab()
    vocab.load_char2id_pkl()
    vocab.reduce_vocabulary(VOCAB_SIZE)

    all_dataloader = get_dataloader_pipeline(vocab, maxlen=MAXLEN, batch_size=BATCH_SIZE)
    train_dataloader = all_dataloader[0]
    val_dataloader = all_dataloader[1]
    test_dataloader = all_dataloader[2]

    # --------------------------------------
    # Modelの作成
    # --------------------------------------
    from models.seq2seq_transform import Seq2Seq

    input_dim = len(vocab.vocab_X.char2id)
    output_dim = len(vocab.vocab_y.char2id)

    model = Seq2Seq(input_dim, output_dim, vocab, maxlen=MAXLEN + 8)

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

    torch.backends.cudnn.benchmark = True
    freeze_support()
    trainer = pl.Trainer(
        strategy="ddp", accelerator="gpu", devices=2, callbacks=callbacks, max_epochs=EPOCH_SIZE
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    train()
