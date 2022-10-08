import os

import dill
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from vocab.twitter_vocab import TwitterVocab


class TwitterDataset(Dataset):
    def __init__(self, messages=None, responses=None, maxlen=80) -> None:
        self.messages = messages
        self.responses = responses
        self.maxlen = maxlen

        if messages is not None and responses is not None:
            err_msg = "発話リストと応答リストのサイズが一致しない．"
            assert len(self.messages) == len(self.responses), err_msg

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index: int):
        if self.messages is None or self.responses is None:
            err_msg = "発話／応答リストがセットされていません．"
            raise ValueError(err_msg)
        if index >= len(self.messages):
            raise StopIteration

        msg = self.messages[index]
        res = self.responses[index]

        return {"source": msg, "target": res}

    def load_dataset_pkl(self, name: str, folder="assets"):
        filepath = os.path.join(folder, name)
        with open(filepath, "rb") as f:
            self.messages = dill.load(f)
            self.responses = dill.load(f)
            self.maxlen = dill.load(f)

    def save_dataset_pkl(self, name: str, folder="assets"):
        filepath = os.path.join(folder, name)
        with open(filepath, "wb") as f:
            dill.dump(self.messages, f)
            dill.dump(self.responses, f)
            dill.dump(self.maxlen, f)


def collate_fn(batch, vocab: TwitterVocab, maxlen: int):
    X = [item["source"] for item in batch]
    y = [item["target"] for item in batch]

    X = vocab.vocab_X.transform(X, is_wakati=False)
    y = vocab.vocab_y.transform(y, is_wakati=False)

    X = [torch.LongTensor(item) for item in X]
    y = [torch.LongTensor(item) for item in y]
    assert len(X) == len(y), "Xとyのサイズが一致しない"

    Xy = pad_sequence(X + y, batch_first=True, padding_value=0)
    X = Xy[: len(X), :]
    y = Xy[len(X) :, :]
    assert X.size() == y.size(), "Xとyのサイズが一致しない"

    X = X.t()
    y = y.t()
    return X, y
