import dill
from collections import Counter
from typing import List
from fugashi import Tagger
from tqdm import tqdm


class Vocab(object):
    def __init__(self, max_vocab=80000) -> None:
        self._words = []
        self.char2id = {}
        self.id2char = {}
        self.max_vocab = max_vocab
        self.parser = Tagger("-Owakati").parse

        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
        self.pad = self.special_tokens[0]
        self.bos = self.special_tokens[1]
        self.eos = self.special_tokens[2]
        self.unk = self.special_tokens[3]

    def _convert_wakati_sentence(self, sentence: str):
        return self.parser(sentence)

    def _set_tokens(self, sentences: List[str], is_wakati: bool):
        for sentence in sentences:
            if not is_wakati:
                sentence = self.parser(sentence)
            for token in sentence.split():
                self._words.append(token)
        return self._words

    def fit(self, sentences: List[str], is_wakati=True, verbose=False):
        if verbose:
            print("語彙からIDへのマップ辞書を更新します...")
            sentences = tqdm(sentences)

        self._words = self._set_tokens(sentences, is_wakati)
        counter = Counter(self._words).most_common(self.max_vocab)
        self._words = [c[0] for c in counter]

        self.char2id = {
            token: len(self.special_tokens) + idx for idx, token in enumerate(self._words)
        }
        for idx, token in enumerate(self.special_tokens):
            self.char2id[token] = idx

        self.id2char = {v: k for k, v in self.char2id.items()}

    def transform(self, sentences: List[str], is_wakati=True, verbose=False, bos=True, eos=True):
        output_t = []

        if verbose:
            sentences = tqdm(sentences)

        for sentence in sentences:
            output_t.append(self.encode(sentence, is_wakati))
        return output_t

    def encode(self, sentence: str, is_wakati: bool, bos=True, eos=True):
        output_e = []

        if not is_wakati:
            sentence = self.parser(sentence)

        for token in sentence.split():
            if token in self.char2id.keys():
                output_e.append(self.char2id[token])
            else:
                output_e.append(self.char2id[self.unk])

        if bos:
            output_e = [self.char2id[self.bos]] + output_e
        if eos:
            output_e = output_e + [self.char2id[self.eos]]

        return output_e

    def convert_ids_to_str(self, batch_id, get_list_str=True):
        """ 
        batch:[List[List[int]]]で受け取るID列バッチを文字列リストに変換する．
        get_list_str: bool = True   トークンリスト:List[str]をjoinでstrに変換する．
        """
        output_c = []
        for id_seq in batch_id:
            output_c.append(self.decode(id_seq, get_list_str))
        return output_c

    def decode(self, id_seq: List[int], get_list_str=True) -> List[str]:
        if get_list_str:
            return "".join([self.id2char[idx] for idx in id_seq])
        else:
            return [self.id2char[idx] for idx in id_seq]

