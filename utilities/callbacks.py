import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class DisplayGeneratedResponses(Callback):
    def __init__(self, vocab) -> None:
        super().__init__()

        self.vocab = vocab
        self._train_batch_list = []

    def __display_responces(self, source, target, preds):
        src = self.vocab.vocab_X.decode(source.tolist())
        tgt = self.vocab.vocab_y.decode(target.tolist())
        pred = self.vocab.vocab_y.decode(preds.argmax(-1).tolist())

        print(f"source : {src}")
        print(f"target : {tgt}")
        print(f"predict : {pred}")
        print("-" * 20)

    def __display_responces_batch(self, batch):
        batch = batch[-1]
        source = [item["source"] for item in batch[-5:-1]]
        target = [item["target"] for item in batch[-5:-1]]
        preds = [item["preds"] for item in batch[-5:-1]]

        for src, tgt, pred in zip(source, target, preds):
            self.__display_responces(src, tgt, pred)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx,
    ) -> None:
        self._train_batch_list.append(outputs)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Generated responses in train data : ")
        self.__display_responces_batch(self._train_batch_list)
        del self._train_batch_list
        self._train_batch_list = []

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

