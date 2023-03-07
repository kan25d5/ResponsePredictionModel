import re
import unicodedata

import emoji
import neologdn

# Twitter固有の不要文字パターン
PATTERN_TWITTER = [
    r"#[^#\s]*",  # ハッシュタグ
    r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+",  # URL
    r"@[a-zA-Z\d_]+",  # スクリーンネーム
]
PATTERN_TWITTER_REPLACE = [("&gt;", ">"), ("&lt;", "<"), ("&amp;", "&"), ("\n", "。")]


class TwitterTransform(object):
    def __init__(self, is_wakati=False) -> None:
        self.re_twitter = [re.compile(r) for r in PATTERN_TWITTER]
        self.is_wakati = is_wakati

    def remove_deburis(self, text: str):
        text = text.replace("�", "")
        for r in self.re_twitter:
            text = r.sub("", text)
        for pair in PATTERN_TWITTER_REPLACE:
            text = text.replace(pair[0], pair[1])
        text = emoji.replace_emoji(text)
        return text

    def _round_period(self, text: str):
        while True:
            if "。。" not in text:
                return text
            text = text.replace("。。", "。")

    def normalize(self, text: str):
        text = self._round_period(text)
        text = text.strip()
        text = text.lower()
        text = neologdn.normalize(text)
        text = unicodedata.normalize("NFKC", text)
        return text

    def preprocess(self, text: str):
        text = self.normalize(text)
        text = self.remove_deburis(text)
        return text

    def __call__(self, text: str):
        return self.preprocess(text)
