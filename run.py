MAXLEN = 120
BATCH_SIZE = 80


def main():
    import os

    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # --------------------------------------
    # Vocab / all_datasetの作成
    # --------------------------------------
    from vocab.twitter_vocab import TwitterVocab
    from utilities.training_functions import get_corpus, get_dataset

    vocab = TwitterVocab()
    vocab.load_char2id_pkl()

    X, y = get_corpus()
    all_dataset = get_dataset(X, y, vocab, maxlen=MAXLEN)

    # --------------------------------------
    # DataLoaderの作成
    # --------------------------------------
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        all_dataset[0],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=all_dataset[0].collate_fn,
        num_workers=52 - 6,
    )
    val_dataloader = DataLoader(
        all_dataset[1],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=all_dataset[1].collate_fn,
        num_workers=52 - 6,
    )
    test_dataloader = DataLoader(
        all_dataset[2],
        batch_size=1,
        shuffle=True,
        collate_fn=all_dataset[2].collate_fn,
        num_workers=52 - 6,
    )

    # --------------------------------------
    # Modelの作成
    # --------------------------------------
    from models.seq2seq_transform import Seq2Seq

    input_dim = len(vocab.vocab_X.char2id)
    output_dim = len(vocab.vocab_y.char2id)

    model = Seq2Seq(input_dim, output_dim, maxlen=MAXLEN + 8)

    # --------------------------------------
    # Modelの適合
    # --------------------------------------
    import pytorch_lightning as pl
    from multiprocessing import freeze_support

    freeze_support()
    trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=2,)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
