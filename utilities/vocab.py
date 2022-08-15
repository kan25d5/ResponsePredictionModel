from typing import List
from collections import Counter


class Vocab(object):
    def __init__(self, top_words=80000) -> None:
        self.top_words = top_words
        self._words = []
        self.char2id = {}
        self.id2char = {}
        self.special_words = ["<pad>", "<s>", "</s>", "<unk>"]
        self.pad_char = self.special_words[0]
        self.bos_char = self.special_words[1]
        self.eos_char = self.special_words[2]
        self.unk_char = self.special_words[3]

    def fit(self, sentences: List[str]):
        for sentence in sentences:
            for word in sentence.split():
                self._words.append(word)

        counter = Counter(self._words).most_common(self.top_words)
        self._words = [c[0] for c in counter]

        self.char2id = {w: (len(self.special_words) + idx) for idx, w in enumerate(self._words)}
        for idx, w in enumerate(self.special_words):
            self.char2id[w] = idx

        self.id2char = {v: k for k, v in self.char2id.items()}

    def transform(self, sentences, bos=True, eos=True):
        output_t = []
        for sentence in sentences:
            output_t.append(self.encode(sentence, bos, eos))
        return output_t

    def encode(self, sentence, bos=True, eos=True):
        output_e = []
        unk_idx = self.char2id[self.unk_char]
        bos_idx = self.char2id[self.bos_char]
        eos_idx = self.char2id[self.eos_char]

        for w in sentence.split():
            if w in self.char2id:
                output_e.append(self.char2id[w])
            else:

                output_e.append(unk_idx)
        if bos:
            output_e = [bos_idx] + output_e
        if eos:
            output_e = output_e + [eos_idx]

        return output_e

    def decode(self, idx_seq):
        return [self.id2char[c] for c in idx_seq]

