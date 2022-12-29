import glob
import os
from multiprocessing import Process, Queue

from tqdm import tqdm

from utilities.transform import TwitterTransform
from utilities.utility_functions import load_json, save_json

NUM_WORKS = os.cpu_count() - 1
CORPUS_FOLDER = "/home/s2110184/dataset/dialog_from_twitter_formatting/*.json"

transform = TwitterTransform()


def split_list(l_, n_):
    return zip(*[iter(l_)] * n_)


def _make_all_corpus_one_process(filepath: str, queue: Queue):
    corpus_msg = []
    corpus_res = []
    corpus = load_json(filepath)

    for dialogue in corpus:
        for i in range(len(dialogue) - 1):
            msg = dialogue[i]["text"]
            res = dialogue[i + 1]["text"]

            msg = transform(msg, use_parser="")
            res = transform(res, use_parser="")

            corpus_msg.append(msg)
            corpus_res.append(res)

    queue.put((corpus_msg, corpus_res))


def _make_all_corpus_multi_process(corpus_split):
    multi_result_msg = []
    multi_result_res = []
    process_list = []
    queue_list = []

    for filepath in corpus_split:
        queue = Queue()
        process = Process(target=_make_all_corpus_one_process, args=(filepath, queue))
        process.start()
        process_list.append(process)
        queue_list.append(queue)

    for process, queue in zip(process_list, queue_list):
        process.join()
        corpus_msg, corpus_res = queue.get()
        multi_result_msg.extend(corpus_msg)
        multi_result_res.extend(corpus_res)

    print("\tcorpus_msg : {}".format(corpus_msg))
    print("\tcorpus_res : {}".format(corpus_res))

    return multi_result_msg, multi_result_res


def make_all_corpus():
    messages = []
    responses = []

    # コーパスファイル全体のリスト
    corpus_file_list = glob.glob(CORPUS_FOLDER)
    # コーパスファイルをNUM_WORKSの数ずつ分割する
    corpus_split_list = list(split_list(corpus_file_list, NUM_WORKS))

    # len(corpus_file_list) / NUM_WORKS 分だけのループ
    for corpus_split in tqdm(corpus_split_list):
        multi_result_msg, multi_result_res = _make_all_corpus_multi_process(
            corpus_split
        )
        messages.extend(multi_result_msg)
        responses.extend(multi_result_res)

        assert len(messages) == len(responses), "発話と応答のリストサイズが不一致。"
        print("Number of turn : {}".format(len(messages)))

    result = {"messages": messages, "responses": responses}
    save_json(result, "assets/corpus/json/all.json")


def make_corpus():
    make_all_corpus()
