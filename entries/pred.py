import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from models.seq2seq_transformer import Seq2SeqTransformer
from utilities.training_functions import (
    collate_batch,
    get_corpus_df,
    get_datasets,
    get_transform,
    load_vocabs,
)
from utilities.utility_functions import init_boilerplate

source_vocab: Vocab
target_vocab: Vocab


def _get_dataloader(df, source_transform, target_transform):
    test_dataloader = DataLoader(
        df.values,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        collate_fn=lambda batch: collate_batch(
            batch, source_transform, target_transform
        ),
    )
    return test_dataloader


def get_predict_pipline():
    global source_vocab
    global target_vocab

    # 各極性のコーパスをロード
    df_pos = get_corpus_df("pos")
    df_neg = get_corpus_df("neg")
    df_neu = get_corpus_df("neu")

    # Source, Targetの
    # 語彙セットクラスtorchtext.vocab.Vocabを作成
    # source_vocab, target_vocabはフィールド変数
    source_vocab, target_vocab = load_vocabs()

    # torchtext.Transformを作成
    source_transform, target_transform = get_transform(source_vocab, target_vocab)

    # 各極性のTestだけを利用する
    dataset_pos = get_datasets(df_pos)[2]
    dataset_neg = get_datasets(df_neg)[2]
    dataset_neu = get_datasets(df_neu)[2]

    # DataLoaderを作成する
    dataloader_pos = _get_dataloader(dataset_pos, source_transform, target_transform)
    dataloader_neg = _get_dataloader(dataset_neg, source_transform, target_transform)
    dataloader_neu = _get_dataloader(dataset_neu, source_transform, target_transform)

    all_dataloader = [dataloader_pos, dataloader_neg, dataloader_neu]
    return all_dataloader


def get_model(args, sentiment_type: str):
    filepath = f"assets/state_dict/{sentiment_type}.pth"

    src_vocab_size = len(source_vocab.get_stoi())
    tgt_vocab_size = len(target_vocab.get_stoi())

    model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size, beam_size=args.beam_size)
    model.load_state_dict(torch.load(filepath), strict=False)

    return model


def display_result(src, tgt, pred, pred2=None):
    src = [item[0] for item in src.tolist() if item[0] != 0]
    tgt = [item[0] for item in tgt.tolist() if item[0] != 0]
    pred = [item[0] for item in pred.tolist() if item[0] != 0]

    src = source_vocab.lookup_tokens(src)
    tgt = target_vocab.lookup_tokens(tgt)
    pred = target_vocab.lookup_tokens(pred)
    result = {"source": "".join(src), "target": "".join(tgt), "pred": "".join(pred)}

    if pred2 is not None:
        pred2 = [item[0] for item in pred2.tolist() if item[0] != 0]
        pred2 = target_vocab.lookup_tokens(pred2)
        result.update({"pred2": "".join(pred2)})

    return result


def display_dataloader(dataloader, model, display_count=10):
    for idx, batch in enumerate(dataloader):
        src, tgt = batch
        pred = model(src).type(torch.LongTensor)

        result = display_result(src, tgt, pred)
        print("source : {}".format(result["source"]))
        print("target : {}".format(result["target"]))
        print("pred : {}".format(result["pred"]))
        print("-" * 20)

        if display_count <= 0 or idx >= display_count:
            break


def display_dataloader_neu(dataloader_neu, model_pos, model_neg, display_count=10):
    for idx, batch in enumerate(dataloader_neu):
        src, tgt = batch
        pred_pos = model_pos(src).type(torch.LongTensor)
        pred_neg = model_neg(src).type(torch.LongTensor)

        result = display_result(src, tgt, pred_pos, pred_neg)
        print("source : {}".format(result["source"]))
        print("target : {}".format(result["target"]))
        print("pred pos : {}".format(result["pred"]))
        print("pred neg : {}".format(result["pred2"]))
        print("-" * 20)

        if display_count <= 0 or idx >= display_count:
            break


def preds(args):
    init_boilerplate(args.devices)
    all_dataloader = get_predict_pipline()
    dataloader_pos = all_dataloader[0]
    dataloader_neg = all_dataloader[1]
    dataloader_neu = all_dataloader[2]

    model_pos = get_model(args, "pos")
    model_neg = get_model(args, "neg")
    model_neu = get_model(args, "neu")

    print("_/" * 30)
    print("まずはpos/neg単体で対話例を生成 : ")
    print("_/" * 30)
    print("load pos model : ")
    print("=" * 30)
    display_dataloader(dataloader_pos, model_pos, display_count=500)
    print("load neg model : ")
    print("=" * 30)
    display_dataloader(dataloader_neg, model_neg, display_count=500)
    print("load neu model : ")
    print("=" * 30)
    display_dataloader(dataloader_neu, model_neu, display_count=500)

    print("\n" * 3)
    print("次に、neuのソースからpos/negモデルでの応答を見る")
    print("=" * 30)
    display_dataloader_neu(dataloader_neu, model_pos, model_neg, display_count=500)
