import json
import os


def load_json(filepath: str):
    with open(filepath) as f:
        json_ = json.load(f)
    return json_


def save_json(content, filepath: str):
    with open(filepath, "w") as f:
        f.write(json.dumps(content, ensure_ascii=False, indent=4))


def init_boilerplate(devices: int):
    """環境変数の設定。モデルをGPU上で動かす際に必要。"""
    # --------------------------------------
    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    # --------------------------------------

    os.environ["MKL_NUM_THREADS"] = "0"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # ------------------------------------
    # Reference:
    # https://github.com/Lightning-AI/lightning/issues/1314#issuecomment-706607614
    # ------------------------------------
    if devices <= 0:
        raise ValueError("デバイスの指定が0以下です．--device={}".format(devices))
    elif devices == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(devices)])
