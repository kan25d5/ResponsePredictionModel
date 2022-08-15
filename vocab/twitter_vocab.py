import dill
from typing import List
from vocab.vocab import Vocab


class TwitterVocab(object):
    def __init__(self, max_vocab=80000,) -> None:
        self.max_vocab = max_vocab
        self.vocab_X = Vocab(max_vocab)
        self.vocab_y = Vocab(max_vocab)

    def fit(self, messages: List[str], responses: List[str], is_wakati=True):
        """
        語彙からIDへのマップ辞書を適合します．

        引数
        -----------------
        - messages: List[str]   発話リスト
        - responses: List[str]    応答リスト
        - is_wakati: bool = True   リストは半角スペースで分かち書きされているか
        """
        # サイズチェック
        err_msg = "発話サイズと応答サイズが一致しない．"
        assert len(messages) == len(responses), err_msg

        # それぞれのVocabクラスで語彙を適合
        print("語彙からIDへのマップ辞書を作成します...")
        print("発話リストを適合：")
        self.vocab_X.fit(messages, is_wakati=is_wakati)
        print("応答リストを適合：")
        self.vocab_y.fit(responses, is_wakati=is_wakati)

    def transform(self, batch):
        """ バッチ化した発話／応答データに対し，ID列変換します． """
        X = [item["source"] for item in batch]
        y = [item["target"] for item in batch]

        X = self.vocab_X.transform(X)
        y = self.vocab_y.transform(y)

        return X, y

    def save_char2id_pkl(self, filepath="assets/char2id.model"):
        """ 語彙からIDへのマップ辞書を保存します． """
        with open(filepath, "wb") as f:
            dill.dump(self.vocab_X.char2id, f)
            dill.dump(self.vocab_y.char2id, f)

    def load_char2id_pkl(self, filepath="assets/char2id.model"):
        """ 語彙からIDへのマップ辞書をロードします． """
        with open(filepath, "rb") as f:
            self.vocab_X.char2id = dill.load(f)
            self.vocab_y.char2id = dill.load(f)

        # ID to token 辞書のロード
        self.vocab_X.id2char = {v: k for k, v in self.vocab_X.char2id.items()}
        self.vocab_y.id2char = {v: k for k, v in self.vocab_y.char2id.items()}

        # 語彙リストのロード
        self.vocab_X._words = list(self.vocab_X.id2char.keys())
        self.vocab_y._words = list(self.vocab_y.id2char.keys())
