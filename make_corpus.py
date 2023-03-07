import glob
import json
import os

import oseti
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utilities.transform import TwitterTransform

analyzer = oseti.Analyzer(mecab_args="-d /home/s2110184/usr/mecab-ipadic-neologd")
transform = TwitterTransform()
sp = spm.SentencePieceProcessor()
sp.Load("assets/japanese-dialog-transformers/data/dicts/sp_oall_32k.model")

CORPUS_FOLDER = "/home/s2110184/dataset/dialog_from_twitter_formatting"


def load_json(filepath: str):
    with open(filepath) as f:
        json_ = json.load(f)
    return json_


def save_json(content, filepath: str):
    with open(filepath, "w") as f:
        f.write(json.dumps(content, ensure_ascii=False, indent=4))


def save_txt(txt_list, filepath: str):
    with open(filepath, "w") as f:
        for line in txt_list:
            f.write(line + "\n")


def tokenizer(raw_txt: str):
    """SentencePieceによる空白トークナイズ"""
    return " ".join(sp.EncodeAsPieces(raw_txt))


def toneinze_raw_txt(folder_path: str):
    """トークナイズをfairseq用に作成したテキストファイルに適用する"""
    for file in glob.glob(folder_path):
        with open(file) as f:
            lines = f.readlines()

        with open(file, "w") as f:
            for line in lines:
                f.write(tokenizer(line) + "\n")


def make_fairseq_rawtxtfiles():
    """twitterコーパスをfairseq用テキストファイルとして作成する"""
    pass


def _make_Xy_from_twitter_corpus(raw_corpus):
    """Twitterコーパスから発話リストと応答リストを返す"""
    X, y = [], []

    for dialogue in raw_corpus:
        for i in range(len(dialogue) - 1):
            msg = dialogue[i]["text"]
            res = dialogue[i + 1]["text"]

            msg = transform(msg)
            res = transform(res)

            X.append(msg)
            y.append(res)

    assert len(X) == len(y), "発話と応答が一致しない．"
    return X, y


def _split_sentimet_corpus(result):
    pos_list = {"X": [], "y": []}
    neg_list = {"X": [], "y": []}
    neu_list = {"X": [], "y": []}

    for i in tqdm(range(len(result["X"]))):
        msg = result["X"][i]
        res = result["y"][i]

        score_list = analyzer.analyze(res)
        if len(score_list) == 0:
            score = 0
        else:
            score = sum(score_list)

        if score == 0:
            neu_list["X"].append(msg)
            neu_list["y"].append(res)
        elif score > 0:
            pos_list["X"].append(msg)
            pos_list["y"].append(res)
        else:
            neg_list["X"].append(msg)
            neg_list["y"].append(res)

    print("result : ")
    print("\t neu len : {}".format(len(neu_list["X"])))
    print("\t pos len : {}".format(len(pos_list["X"])))
    print("\t neg len : {}".format(len(neg_list["X"])))

    result = {"neu": neu_list, "neg": neg_list, "pos": pos_list}
    return result


def _make_all_corpus():
    messages = []
    responses = []

    for file in tqdm(glob.glob(os.path.join(CORPUS_FOLDER, "*.json"))):
        corpus = load_json(file)
        X, y = _make_Xy_from_twitter_corpus(corpus)
        messages.extend(X)
        responses.extend(y)

    assert len(messages) == len(responses), "発話と応答が一致しない．"
    result = {"X": messages, "y": responses}

    return result


def make_fairseq_txt(corpus, corpus_type):
    X, y = corpus["X"], corpus["y"]
    assert len(X) == len(y), "サイズが一致しない．"

    # 2文字以下の発話が含まれるターンを削除
    i = 0
    dis_count = 0
    while i <= (len(X) - dis_count):
        if len(X[i]) <= 2 or len(y[i]) <= 2:
            X.pop(i)
            y.pop(i)
            dis_count += 1
        i += 1

    # train/val/testに分割
    X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=0.7)
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other)

    corpus_list = [
        (X_train, y_train, "train"),
        (X_val, y_val, "valid"),
        (X_test, y_test, "test"),
    ]

    # テキストファイルを作成する
    for corpus_line in corpus_list:
        X_lines, y_lines, label = corpus_line
        f_src = open(f"assets/corpus/faitseq/raw/{corpus_type}/{label}.src", "w")
        f_dst = open(f"assets/corpus/faitseq/raw/{corpus_type}/{label}.dst", "w")

        for x, y in zip(X_lines, y_lines):
            f_src.write(tokenizer(x) + "\n")
            f_dst.write(tokenizer(y) + "\n")

        f_src.close()
        f_dst.close()


def make_sentiment_corpus():
    """twitterコーパスを極性解析して，sentimetcorpusを作成する"""
    print("all.jsonの作成：")
    all_result = _make_all_corpus()
    save_json(all_result, "fairseq_script/temp/all.json")

    print("sentiment corpusの作成：")
    sent_result = _split_sentimet_corpus(all_result)
    save_json(sent_result["neu"], "fairseq_script/temp/neu.json")
    save_json(sent_result["pos"], "fairseq_script/temp/pos.json")
    save_json(sent_result["neg"], "fairseq_script/temp/neg.json")

    for label in ["neu", "pos", "neg"]:
        corpus = load_json(f"assets/corpus/temp/{label}.json")
        make_fairseq_txt(corpus, label)


if __name__ == "__main__":
    make_sentiment_corpus()
