import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class DisplayGeneratedResponses(Callback):
    def __init__(self, vocab, train_dataloader) -> None:
        super().__init__()

        self.vocab = vocab
        self.train_dataloader = train_dataloader

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Generated responses in train data : ")
        for idx, (x, t) in enumerate(self.train_dataloader):
            pred = pl_module(x.to("cuda")).to("cpu")
            self.__display_responces(x, t, pred)
            if idx >= 5:
                break

    def __display_responces(self, source, target, preds):
        src = self.vocab.vocab_X.decode(source.tolist())
        tgt = self.vocab.vocab_y.decode(target.tolist())
        pred = self.vocab.vocab_y.decode(preds.argmax(-1).tolist())

        print(f"\tsource : {src}")
        print(f"\ttarget : {tgt}")
        print(f"\tpredict : {pred}")
        print("-" * 20)

    def __display_responces_batch(self, batch):
        batch = batch[-1]
        source = [item["source"] for item in batch[-5:-1]]
        target = [item["target"] for item in batch[-5:-1]]
        preds = [item["preds"] for item in batch[-5:-1]]

        for src, tgt, pred in zip(source, target, preds):
            self.__display_responces(src, tgt, pred)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        if trainer.current_epoch > 0:
            print("Generated responses in val data : ")
            self.__display_responces_batch(outputs)

