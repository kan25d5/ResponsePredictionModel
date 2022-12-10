import os

import pandas as pd
import torch
import torchtext.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from utilities.transform import TwitterTransform
from utilities.utility_functions import load_json

# def json_to_tsv():
#     for filepath in glob.glob("assets/corpus/json/*.json"):
#         corpus = load_json(filepath)
#         with open(filepath.replace("json", "tsv"), "w") as f:
#             writer = csv.writer(f, delimiter="\t")
#             writer.writerow(["source", "target"])
#             for msg, res in zip(corpus["X"], corpus["y"]):
#                 writer.writerow([msg, res])


def get_corpus_df(corpus_type: str, tokenizer=None):
    """コーパスのタイプを指定して，DataFrameで返す

    Returns:
        df(pd.DataFrame)
    """
    # コーパスをロードする
    df = pd.read_csv(f"assets/corpus/tsv/{corpus_type}.tsv", sep="\t")
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    # ロードしたコーパスを分かち書きする
    if tokenizer is None:
        tokenizer = TwitterTransform(is_wakati=True)

    tokenizer = get_tokenizer(tokenizer=tokenizer, language="ja")
    df["source"] = df["source"].map(lambda x: tokenizer(x).split())
    df["target"] = df["target"].map(lambda x: tokenizer(x).split())
    return df


def make_vocabs(
    df: pd.DataFrame,
    vocab_size: int,
    is_saved=True,
    source_filename="source_vocab.pth",
    target_filename="target_vocab.pth",
):
    """データセットから単語を取り出しtorchtext.vocab.Vocabを作成する．
    https://pytorch.org/text/main/vocab.html#torchtext.vocab.Vocab
    """
    # コーパスからVocabオブジェクトを作成する
    source_vocab = build_vocab_from_iterator(
        df["source"], specials=("<pad>", "<bos>", "<eos>", "<unk>"), max_tokens=vocab_size
    )
    target_vocab = build_vocab_from_iterator(
        df["target"], specials=("<pad>", "<bos>", "<eos>", "<unk>"), max_tokens=vocab_size
    )

    # デフォルトインデックスを<unk>に設定
    source_vocab.set_default_index(source_vocab["<unk>"])
    target_vocab.set_default_index(target_vocab["<unk>"])

    if is_saved:
        torch.save(source_vocab, os.path.join("assets/vocab", source_filename))
        torch.save(target_vocab, os.path.join("assets/vocab", target_filename))

    return source_vocab, target_vocab


def load_vocabs(
    source_filename="source_vocab.pth",
    target_filename="target_vocab.pth",
):
    """作成した辞書データをロードする"""
    source_vocab = torch.load(os.path.join("assets/vocab", source_filename))
    target_vocab = torch.load(os.path.join("assets/vocab", target_filename))
    return source_vocab, target_vocab


def get_transform(source_vocab, target_vocab):
    """torchtext.Transformを作成する．
        https://pytorch.org/text/main/transforms.html

    Args:
        source_vocab (torchtext.vocab.Vocab): SourceのVocab
        target_vocab (torchtext.vocab.Vocab): targetのVocab

    Returns:
        source_transform, target_transform: SourceとTargetのTransform
    """

    # transformの設定
    # ref : https://pytorch.org/text/main/transforms.html
    source_transform = T.Sequential(
        T.VocabTransform(source_vocab),  # トークン-to-インデックスの設定
        T.AddToken(token=source_vocab["<bos>"], begin=True),  # BOSトークン
        T.AddToken(token=source_vocab["<eos>"], begin=False),  # EOSトークン
        T.ToTensor(padding_value=source_vocab["<pad>"]),  # パディング処理
    )
    target_transform = T.Sequential(
        T.VocabTransform(target_vocab),  # トークン-to-インデックスの設定
        T.AddToken(token=target_vocab["<bos>"], begin=True),  # BOSトークン
        T.AddToken(token=target_vocab["<eos>"], begin=False),  # EOSトークン
        T.ToTensor(padding_value=target_vocab["<pad>"]),  # パディング処理
    )

    return source_transform, target_transform


def get_datasets(df: pd.DataFrame):
    """データセットを訓練/検証/テストの3つに分割する．

    Args:
        df (pd.DataFrame): get_corpus_df()の戻り値

    Returns:
        [train_dataset, val_dataset, test_dataset]: 分割したデータセットのpd.DataFrameである．
        {"source" : [source_sequence], "target:[target_sequence]}のようになっている．
        各_sequenceはトークンリスト．
    """

    # データセットを訓練/検証/testに分割する
    X_train, X_other, y_train, y_other = train_test_split(df["source"], df["target"])
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other)

    # 各データのXとyをデータをひとまとめにする
    train_dataset = pd.DataFrame({"source": X_train, "target": y_train})
    val_dataset = pd.DataFrame({"source": X_val, "target": y_val})
    test_dataset = pd.DataFrame({"source": X_test, "target": y_test})

    all_dataset = [train_dataset, val_dataset, test_dataset]
    return all_dataset


def collate_batch(batch, source_transform, target_transform):
    sources = source_transform([source for (source, target) in batch])
    targets = target_transform([target for (source, target) in batch])
    return sources.t(), targets.t()


def get_dataloader(
    all_dataset, source_transform, target_transform, batch_size: int, num_workers=8
):
    train_dataset = all_dataset[0]
    val_dataset = all_dataset[1]
    test_dataset = all_dataset[2]

    # データローダーを作成する
    train_dataloader = DataLoader(
        train_dataset.values,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, source_transform, target_transform),
    )
    val_dataloader = DataLoader(
        val_dataset.values,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, source_transform, target_transform),
    )
    test_dataloader = DataLoader(
        test_dataset.values,
        batch_size=1,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, source_transform, target_transform),
    )
    test_callback_dataloader = DataLoader(
        test_dataset.values,
        batch_size=1,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, source_transform, target_transform),
    )
    all_dataloader = [train_dataloader, val_dataloader, test_dataloader, test_callback_dataloader]

    return all_dataloader
