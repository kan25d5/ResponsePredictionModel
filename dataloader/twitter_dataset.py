import torch
from typing import List
from torch.utils.data import Dataset
from vocab.twitter_vocab import TwitterVocab
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TwitterDataset(Dataset):
    def __init__(
        self, messages, responses, vocab: TwitterVocab, transform=None, maxlen=80
    ) -> None:
        self.messages = messages
        self.responses = responses
        self.vocab = vocab
        self.maxlen = maxlen
        self.transform = transform

        err_msg = "発話リストと応答リストのサイズが一致しない．"
        assert len(self.messages) == len(self.responses), err_msg

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index: int):
        if index >= len(self.messages):
            raise StopIteration

        msg = self.messages[index]
        res = self.responses[index]

        msg = self.vocab.vocab_X.encode(msg)
        res = self.vocab.vocab_y.encode(res)

        if self.transform is not None:
            msg = self.transform(msg)
            res = self.transform(res)

        return {"source": msg, "target": res}

    def collate_fn(self, batch):
        X = [item["source"] for item in batch]
        y = [item["target"] for item in batch]

        X = pad_sequences(X, maxlen=self.maxlen, padding="post")
        y = pad_sequences(y, maxlen=self.maxlen, padding="post")

        X = torch.LongTensor(X).t()
        y = torch.LongTensor(y).t()

        return X, y
