import json
import os
from glob import glob
from multiprocessing import Process, Queue
from typing import List

from tqdm import tqdm

from utilities.transform import TwitterTransform
from utilities.utility_functions import load_json, save_json

CORPUS_FOLDER = "/home/s2110184/dataset/dialog_from_twitter_formatting/*.json"
NUM_WORKER = os.cpu_count() - 1
MIN_LEN = 3
MAX_LEN = 120

# Class for text preprocessing
transform = TwitterTransform()


def split_list(l_, n_):
    """Divide the list(l_) by N(n_) elements each."""
    return list(zip(*[iter(l_)] * n_))


def _get_corpus(filepath: str, queue: Queue = None):
    """コーパスファイルをロードして、コーパス中の発話リスト、応答リストを返す。
    load dialogue corpus file, and return messagelist and response list

    Args:
        filepath (str): コーパスファイルパス. corpus file path.
        queue (Queue, optional): queue. Defaults to None.

    Return:
        result (corpus_msg, corpus_res): 発話リストと応答リストのタプル。
            Tuple of message list and response list.
    """

    # message and response list in the
    #   dialogue corpus file.
    corpus_msg = []
    corpus_res = []

    try:
        corpus = load_json(filepath)
    except json.JSONDecodeError:
        if queue is None:
            return (corpus_msg, corpus_res)
        else:
            queue.put((corpus_msg, corpus_res))

    if len(corpus) <= 0:
        if queue is None:
            return (corpus_msg, corpus_res)
        else:
            queue.put((corpus_msg, corpus_res))

    for dialogue in corpus:
        if len(dialogue) <= 0:
            continue

        for i in range(len(dialogue) - 1):
            # pop the message and response
            msg = dialogue[i]["text"]
            res = dialogue[i + 1]["text"]

            # preprocess text (remove debris chars etc).
            msg = transform(msg, use_parser="")
            res = transform(res, use_parser="")

            if len(msg) <= MIN_LEN or len(res) <= MIN_LEN:
                continue
            if len(msg) > MAX_LEN or len(res) > MAX_LEN:
                continue

            # set the message and response in the each list
            corpus_msg.append(msg)
            corpus_res.append(res)

        # the list size of message and response dont match
        assert len(corpus_msg) == len(corpus_res), "発話と応答のリストサイズが一致しない。"

    if queue is None:
        return (corpus_msg, corpus_res)
    else:
        queue.put((corpus_msg, corpus_res))


def make_all_corpus():
    i = 0
    all_turns_count = 0
    corpus_files = list(glob(CORPUS_FOLDER))
    corpus_split_files = split_list(corpus_files, NUM_WORKER)

    for corpus_split in tqdm(corpus_split_files):
        messages = []
        responses = []

        for idx, filepath in enumerate(corpus_split):
            print("transform in process {} / {}".format(idx, len(corpus_split)))
            print("\tloading {}".format(filepath))

            result = _get_corpus(filepath)
            corpus_msg, corpus_res = result

            messages.extend(corpus_msg)
            responses.extend(corpus_res)
            assert len(messages) == len(responses), "発話リストと応答リストサイズが一致しない。"
            print("\tturn : {}".format(len(messages)))
            all_turns_count += len(messages)

        result = {"X": messages, "y": responses}
        filepath = f"assets/corpus/json/all/{i}.json"
        save_json(result, filepath)
        i += 1


def make_corpus():
    make_all_corpus()
