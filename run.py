# --------------------------------------
# デフォルト値の設定
# --------------------------------------
MAXLEN = 80
BATCH_SIZE = 80
EPOCH_SIZE = 200
VOCAB_SIZE = 80000
N_TRIALS = 100
DATA_SIZE = 1.0

SENTIMENT_TYPE = "neu"
STRATEGY = "ddp"
ACCELERATOR = "gpu"
DEVICES = 2
NUM_WORKER = 26
PIN_MEMORY = False
PATIENCE = 3
BEAM_SIZE = 20

PARAMS_PATH = "assets/best_models/best_params.json"


# --------------------------------------
# ArgumentParserの設定
# --------------------------------------
import argparse

# パーサーの作成
description = "積極的／消極的な応答を生成するモデル"
parser = argparse.ArgumentParser(description=description)

# ヘルプの定義
help_mode = "起動するモードを選択する．\
    train : 指定したパラメータでモデルの訓練する．\
    pred : 学習済みのモデルを利用してコマンドライン\
    optuna : ハイパーパラメータを探索する．\
    make_corpus : 語彙データを作成する．\
    load_corpus : 語彙データを確認する．"
help_sentimet = "応答の極性を選択する．\
    pos : 積極的な応答を訓練/生成する．\
    neg : 消極的な応答を訓練/生成する．\
    neural : ニュートラルな応答を訓練/生成する．\
    normal : 全ての応答で訓練する．"
help_devices = "GPUデバイスの数を指定．"
help_maxlen = "応答する系列の最大サイズ．default={}".format(MAXLEN)
help_batch_size = "バッチサイズ．default={}".format(BATCH_SIZE)
help_max_epoch = "最大エポックサイズ．default={}".format(EPOCH_SIZE)
help_vocab_size = "語彙サイズ．default={}".format(VOCAB_SIZE)
help_params = "ハイパーパラメーター値の指定．jsonファイルへのパス．\
    pe_dropout, encoder_dropout, decoder_dropout, learning_ratio \
    encoder_num_layers, decoder_num_layersをDICT形式で指定．"

# コマンドライン引数の追加
parser.add_argument("mode", help=help_mode, type=str)
parser.add_argument("--devices", default=DEVICES, type=int, help=help_devices)
parser.add_argument(
    "-st", "--sentiment_type", help=help_sentimet, type=str, default=SENTIMENT_TYPE
)
parser.add_argument("-len", "--maxlen", help=help_maxlen, type=int, default=MAXLEN)
parser.add_argument(
    "-bt", "--batch_size", help=help_batch_size, type=int, default=BATCH_SIZE
)
parser.add_argument(
    "-ep", "--max_epoch", help=help_max_epoch, type=int, default=EPOCH_SIZE
)
parser.add_argument(
    "-vs", "--vocab_size", help=help_vocab_size, type=int, default=VOCAB_SIZE
)
parser.add_argument("-lr", "--learning_ratio", type=float, default=1e-5)
parser.add_argument("--strategy", type=str, default=STRATEGY)
parser.add_argument("--accelerator", type=str, default=ACCELERATOR)
parser.add_argument("--num_worker", type=int, default=NUM_WORKER)
parser.add_argument("--n_trials", type=int, default=N_TRIALS)
parser.add_argument("--patience", type=int, default=PATIENCE)
parser.add_argument("--beam_size", type=int, default=BEAM_SIZE)
parser.add_argument("--data_size", type=float, default=DATA_SIZE)
parser.add_argument("--params", type=str, default=PARAMS_PATH, help=help_params)
parser.add_argument("--make_vocab", action="store_true")


def main():
    args = parser.parse_args()
    run_mode = args.mode

    if run_mode == "train":
        from entries.train import train

        train(args)
    elif run_mode == "pred":
        from entries.pred import preds

        preds(args)
    elif run_mode == "make_vocab":
        from entries.make_vocab import make_vocab

        make_vocab(args)
    elif run_mode == "load_vocab":
        from entries.make_vocab import load_vocab

        load_vocab()
    elif run_mode == "make_corpus":
        from entries.make_corpus import make_corpus

        make_corpus()
    else:
        raise ValueError("modeの引数が不正．--helpを参照．")


if __name__ == "__main__":
    main()
