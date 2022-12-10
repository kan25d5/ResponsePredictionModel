import random

import pandas as pd

from utilities.training_functions import (get_corpus_df, load_vocabs,
                                          make_vocabs)


def _check_vocab(vocab):
    for i in range(10):
        print("v[{}]:{}  ".format(i, vocab.lookup_token(i)),end="")
    print()
    for _ in range(10):
        i = random.randint(0, len(vocab.get_stoi()) - 1)
        print("v[{}]:{}  ".format(i, vocab.lookup_token(i)), end="")
    print()


def make_vocab(args):
    # コーパスをロード
    df = pd.concat([get_corpus_df("persona"), get_corpus_df("normal")])
    # 辞書データを作成し，セーブする
    source_vocab, target_vocab = make_vocabs(
        df,
        args.vocab_size,
        is_saved=True,
        # ファイル名の指定は場当たり的対応．後で必ず訂正
        source_filename="source_vocab.pth",
        target_filename="target_vocab.pth",
    )

    # 語彙データを確認する
    print("source_vocabs : ")
    _check_vocab(source_vocab)
    print("target_vocab : ")
    _check_vocab(target_vocab)


def load_vocab():
    source_vocab, target_vocab = load_vocabs()

    # 語彙データを確認する
    print("source_vocabs : ")
    _check_vocab(source_vocab)
    print("target_vocab : ")
    _check_vocab(target_vocab)

    print("語彙データを確認する")
