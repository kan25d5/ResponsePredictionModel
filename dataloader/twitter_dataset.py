from typing import List
from torch.utils.data import Dataset
from vocab.twitter_vocab import TwitterVocab


class TwitterDataset(Dataset):
    def __init__(
        self, messages, responses, vocab: TwitterVocab, transform=None
    ) -> None:
        self.messages = messages
        self.responses = responses
        self.vocab = vocab
        self.transform = transform

        err_msg = "発話リストと応答リストのサイズが一致しない．"
        assert len(self.messages) == len(self.responses), err_msg

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index: int):
        msg = self.messages[index]
        res = self.responses[index]

        if self.transform is not None:
            msg = self.transform(msg)
            res = self.transform(res)

        return {"source": msg, "target": res}

    def collate_fn(self, batch):
        return self.vocab.transform(batch)
