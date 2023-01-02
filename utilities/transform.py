import re
import unicodedata

import emoji
import MeCab
import neologdn

from utilities.utility_functions import load_json

# Twitter固有の不要文字パターン
PATTERN_TWITTER = [
    r"#[^#\s]*",  # ハッシュタグ
    r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+",  # URL
    r"@[a-zA-Z\d_]+",  # スクリーンネーム
]
# 顔文字パターン
PATTERN_KAOMOJI = [
    r"\(.*?\)",
    r"([\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF])",
    r"<[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]>",
]
# 特殊顔文字
SPECIAL_KAOMOJI = ["(・ᄇ・)و̑̑", "&lt;(_　_)&gt"]
# 不要文字列
REMOVE_CHARS = [
    "・゜・。",
    "٩",
    "و",
    "|ω・)",
    "pq",
    "σ",
    "|ω・`)",
    "=≡σつ",
    "__",
    ")))",
    "*˘ ³˘)〜♥",
    "ε｀　)",
    "▄【┻┳══ー",
    "⊂二二二二⊃",
    "(*'ω'ノノ゙☆",
    "ς",
    "↜⃕ψ",
    "!!⊃≡゚゚",
    "∵.∵",
    "↜⃕",
    "ヽノ",
    "=3=3=3",
    "≡≡≡≡≡ヘノニゲロ!!",
    "ノ)`-')\"",
    "-o)`з́)-o)`з́)",
    "-o)`з́)",
    "‧o·‧o·̊└┘",
    "。゚゚。",
]
# 想定顔文字数
KAOMOJI_LENS = [4, 5, 6, 7]
# 顔文字の手
KAOMOJI_HANDS = ["ノ", "ヽ", "∑", "m", "O", "o", "┐", "/", "\\", "┌", "٩", "و", "p", "q"]
# MeCab
MECAB_TAGGER = "-Owakati -d/home/s2110184/usr/mecab-ipadic-neologd"
# 顔文字辞書
KAOMOJI_DICT = "/home/s2110184/project/ResponsePredictionModel/assets/dict/kaomoji.json"


class TwitterTransform(object):
    def __init__(self, is_wakati=False) -> None:
        self.is_wakati = is_wakati
        self.parse = MeCab.Tagger(MECAB_TAGGER).parse
        self.pattern_twitter = [re.compile(r) for r in PATTERN_TWITTER]
        self.pattern_kaomoji = [re.compile(r) for r in PATTERN_KAOMOJI]
        self.kaomoji_dic = load_json(KAOMOJI_DICT)

    def remove_deburis(self, text: str):
        for kaomoji in self.kaomoji_dic:
            text = text.replace(kaomoji, "")
        for kaomoji in self.pattern_kaomoji:
            text = kaomoji.sub("", text)
        for twitter in self.pattern_twitter:
            text = twitter.sub("", text)
        for c in REMOVE_CHARS:
            text = text.replace(c, "")
        text = emoji.replace_emoji(text)
        return text

    def normalize(self, text: str):
        text = "".join(text.split())
        text = text.strip()
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        return text

    def wakati(self, text):
        if self.is_wakati:
            text = self.parse(text)
        return text

    def preprocess(self, text: str):
        text = text.replace("�", "")
        text = self.normalize(text)
        text = self.remove_deburis(text)
        text = neologdn.normalize(text)
        text = self.wakati(text)
        return text

    def __call__(self, text: str):
        return self.preprocess(text)
