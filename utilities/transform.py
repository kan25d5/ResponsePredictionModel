import re
import unicodedata

import emoji
import neologdn
from MeCab import Tagger

USER_DICT = "/home/s2110184/opt/mecab/lib/mecab/dic/mecab-ipadic-neologd"

# ref :
# https://qiita.com/sanma_ow/items/b49b39ad5699bbcac0e9
RE_PATTERNS = [
    r"#[^#\s]*",
    r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+",
    r"@[a-zA-Z\d_]+",
    r"\(.*?\)",
    r"（.*?）",
    r"＜.*?＞",
    r"\([^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+?\)",
]
REMOVE_CHARS = ["�", "\n"]


class TwitterTransform(object):
    def __init__(self, is_wakati=False) -> None:
        self.is_wakati = is_wakati
        self.tagger = Tagger("-Owakati -d {}".format(USER_DICT))
        self.re_removes = [re.compile(pattern) for pattern in RE_PATTERNS]

    def __call__(self, text: str):
        for rc in REMOVE_CHARS:
            text = text.replace(rc, "")
        for re_remove in self.re_removes:
            text = re_remove.sub("", text)

        text = text.strip()
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        text = neologdn.normalize(text)
        text = emoji.replace_emoji(text, replace="")

        if self.is_wakati:
            text = self.tagger.parse(text)
        return text
