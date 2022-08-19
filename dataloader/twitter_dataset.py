import os
import dill
import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences


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

    def collate_fn(self, batch):
        X = [item["source"] for item in batch]
        y = [item["target"] for item in batch]

        X = pad_sequences(X, maxlen=self.maxlen, padding="post")
        y = pad_sequences(y, maxlen=self.maxlen, padding="post")

        X = torch.LongTensor(X).t()
        y = torch.LongTensor(y).t()

        return X, y
