import re
import unicodedata

import emoji
import nagisa
import neologdn

# MeCabは利用しない方針に
# ipadic-neologdが大きすぎて解析に
# from assets.constant import USER_DICT

# from MeCab import Tagger


# 顔文字の想定サイズ
KAOMOJI_LEN_LIST = [4, 5, 6, 7, 8]
# 想定される顔文字の「手」に該当する文字
KAOMOJI_HANDS = ["ノ", "ヽ", "∑", "m", "O", "o", "┐", "/", "\\", "┌"]

# ref :
# https://qiita.com/sanma_ow/items/b49b39ad5699bbcac0e9
RE_PATTERNS = [
    r"#[^#\s]*",
    r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+",
    r"@[a-zA-Z\d_]+",
    r"\(.*?\)",
    r"(.*?)",
    r"<.*?>",
    r"＜.*?＞",
    r"\([^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+?\)",
]
REMOVE_CHARS = ["�"]


class TwitterTransform(object):
    def __init__(self, is_wakati=False) -> None:
        self.is_wakati = is_wakati
        # self.parser = Tagger("-Owakati -d {}".format(USER_DICT)).parse
        self.remove_patterns = [re.compile(r) for r in REMOVE_CHARS]

    def _remove_char(self, text: str):
        """不要な文字列の削除"""
        # nagisaによるフィルタリング
        words = nagisa.filter(text, filter_postags=["URL"]).words
        text = "".join(words)
        # 事前に定義したパターンを削除
        for r in self.remove_patterns:
            text = r.sub("", text)
        # 事前に定義した文字列を削除
        for r in REMOVE_CHARS:
            text = text.replace(r, "")
        # 絵文字の削除
        text = emoji.replace_emoji(text)
        return text

    def _extract_kaomoji(self, text: str):
        """顔文字の削除"""
        results = nagisa.extract(text, extract_postags=["補助記号"])
        words = results.words
        kaomoji_words = []

        for kaomoji_len in KAOMOJI_LEN_LIST:
            kaomoji_idx = [i for i, w in enumerate(words) if len(w) >= kaomoji_len]
            for i in kaomoji_idx:
                kaomoji = words[i]
                if len(words) <= i:
                    continue
                if words[i - 1] in KAOMOJI_HANDS and 0 < i:
                    kaomoji = words[i - 1] + kaomoji
                if words[i + 1] in KAOMOJI_HANDS:
                    kaomoji = kaomoji + words[i + 1]
                kaomoji_words.append(kaomoji)
        return kaomoji_words

    def _normalize(self, text: str):
        """文字列の正規化"""
        text = text.strip()
        text = text.lower()
        text = neologdn.normalize(text, repeat=5)
        return text

    def transform(self, text: str):
        # NFKCによるUnicode正規化
        text = unicodedata.normalize("NFKC", text)
        # 顔文字の削除
        for kaomoji in self._extract_kaomoji(text):
            text = text.replace(kaomoji, "")
        # 正規化
        text = self._normalize(text)
        # 不要文字削除
        text = self._remove_char(text)
        # 改行を。に変換
        #   neologdn.normalize(text, repeat=5)で連続文字を5文字に収めている
        text = text.replace("\n\n\n\n\n", "。")
        text = text.replace("\n\n\n\n", "。")
        text = text.replace("\n\n\n", "。")
        text = text.replace("\n\n", "。")
        text = text.replace("\n", "。")
        return text

    def wakati_use_mecab(self, text):
        return self.parser(text)

    def wakati_use_nagisa(self, text):
        return nagisa.tagging(text).words

    def transform_and_wakati(self, text: str, use_parser="nagisa"):
        text = self.transform(text)
        # if use_parser == "mecab":
        #     text = self.wakati_use_mecab(text)
        #     text = text.replace("\n", "")
        if use_parser == "nagisa":
            text = self.wakati_use_nagisa(text)
        elif use_parser == "" or use_parser is None:
            pass
        else:
            # ValueError("use_parserの値はmecab/nagisa/空文字に限る。")
            ValueError("use_parserの値はnagisa/空文字に限る。")
        return text

    def __call__(self, text: str, use_parser="nagisa"):
        self.transform_and_wakati(text, use_parser)
