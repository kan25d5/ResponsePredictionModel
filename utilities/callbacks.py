from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torchtext.vocab import Vocab


class DisplayPredictedResponse(Callback):
    def __init__(
        self, source_vocab: Vocab, target_vocab: Vocab, test_callback_dataloader, display_count=10
    ) -> None:
        super().__init__()
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.display_count = display_count
        self.test_callback_dataloader = test_callback_dataloader

    def get_predited_response(self, model):
        result_list = []

        for idx, (src, tgt) in enumerate(self.test_callback_dataloader):
            if idx >= self.display_count:
                break

            pred = model(src.to("cuda")).to("cpu").type(torch.LongTensor)

            src = [item[0] for item in src.tolist() if item[0] != 0]
            tgt = [item[0] for item in tgt.tolist() if item[0] != 0]
            pred = [item[0] for item in pred.tolist() if item[0] != 0]

            src = self.source_vocab.lookup_tokens(src)
            tgt = self.target_vocab.lookup_tokens(tgt)
            pred = self.target_vocab.lookup_tokens(pred)

            result = {"source": "".join(src), "target": "".join(tgt), "pred": "".join(pred)}
            result_list.append(result)
        return result_list

    def display_predicted_response(self, result: List[dict]):
        for item in result:
            for key, value in item.items():
                print("{} : {}".format(key, value))
            print("-" * 20)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
        result = self.get_predited_response(model)
        self.display_predicted_response(result)
