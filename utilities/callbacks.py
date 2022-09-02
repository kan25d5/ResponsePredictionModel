import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
from vocab.twitter_vocab import TwitterVocab


class DisplaySystenResponses(Callback):
    def __init__(
        self,
        vocab: TwitterVocab,
        dataloader_train_callback: DataLoader,
        dataloader_test: DataLoader,
        filename="base",
    ) -> None:
        super().__init__()

        self.vocab = vocab
        self.filename = filename
        self.dataloader_train_callback = dataloader_train_callback
        self.dataloader_test = dataloader_test

    def display_responses(self, dataloader, pl_module):
        results = []

        for idx, (src, tgt) in enumerate(dataloader):
            pred = pl_module(src.to("cuda")).to("cpu")

            src = [item[0] for item in src.tolist() if item[0] != 0]
            tgt = [item[0] for item in tgt.tolist() if item[0] != 0]
            pred = [item[0] for item in pred.tolist() if item[0] != 0]

            result = {
                "source": self.vocab.vocab_X.decode(src[1:-1]),
                "target": self.vocab.vocab_y.decode(tgt[1:-1]),
                "pred": self.vocab.vocab_y.decode(pred[1:-1]),
            }
            results.append(result)

            print("発話：{}".format(result["source"]))
            print("教師応答：{}".format(result["target"]))
            print("システム応答：{}".format(result["pred"]))
            print("-" * 20)

            if idx >= 8:
                break
        print("=" * 20)

        return results

    def save_txt(self, results, trainer, data_type: str):
        f = open("assets/" + self.filename, "a")

        f.write(f"{trainer.current_epoch} epochs.\n")
        f.write(f"data type {data_type}.\n")
        for result in results:
            f.write("source : {}\n".format(result["source"]))
            f.write("target : {}\n".format(result["target"]))
            f.write("pred : {}\n".format(result["pred"]))

        f.write("============================================\n")
        f.close()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f"\n\n{trainer.current_epoch}エポックにおける，訓練データ中の応答生成結果:\n")
        results = self.display_responses(self.dataloader_train_callback, pl_module)
        self.save_txt(results, trainer, "train")

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        print(f"\n\n{trainer.current_epoch}エポックにおける，検証データ中の応答生成結果:\n")
        results = self.display_responses(self.dataloader_test, pl_module)
        self.save_txt(results, trainer, "val")
