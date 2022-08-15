import dill
from typing import List
from utilities.vocab import Vocab


class TwitterVocab(object):
    def __init__(self, top_words: int = 80000) -> None:
        self.top_words = top_words
        self.vocab_x = Vocab(top_words)
        self.vocab_y = Vocab(top_words)

    def fit(self, msg: List[str], res: List[str]):
        self.vocab_x.fit(msg)
        self.vocab_y.fit(res)

    def transform(self, msg, res, X_bos=True, X_eos=True, y_bos=True, y_eos=True):
        msg = self.vocab_x.transform(msg, X_bos, X_eos)
        res = self.vocab_y.transform(res, y_bos, y_eos)
        return msg, res

    def fit_transform(self, msg, res, X_bos=True, X_eos=True, y_bos=True, y_eos=True):
        self.fit(msg, res)
        msg, res = self.transform(msg, res, X_bos, X_eos, y_bos, y_eos)
        return msg, res

    def load_char2id(self, filepath):
        with open(filepath, "rb") as f:
            self.vocab_x.char2id = dill.load(f)
            self.vocab_y.char2id = dill.load(f)
        self.vocab_x.id2char = {v: k for k, v in self.vocab_x.char2id.items()}
        self.vocab_y.id2char = {v: k for k, v in self.vocab_y.char2id.items()}

    def save_char2id(self, filepath):
        with open(filepath, "wb") as f:
            dill.dump(self.vocab_x.char2id, f)
            dill.dump(self.vocab_y.char2id, f)
