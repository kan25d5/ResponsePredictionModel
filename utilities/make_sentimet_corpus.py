from oseti import Analyzer
from utilities.training_functions import get_corpus
from tqdm import tqdm

from .utility_functions import save_json

analyze = Analyzer().analyze


def main():
    messages, responses = get_corpus()

    err_msg = "発話サイズと応答サイズが一致しない．"
    assert len(messages) == len(responses), err_msg

    results = {
        "pos": {"X": [], "y": []},
        "neg": {"X": [], "y": []},
        "neu": {"X": [], "y": []},
    }

    for msg, res in tqdm(zip(messages, responses), total=len(messages)):
        score = sum(analyze(res))
        if score > 0:
            results["pos"]["X"].append(msg)
            results["pos"]["y"].append(res)
        elif score < 0:
            results["neg"]["X"].append(msg)
            results["neg"]["y"].append(res)
        else:
            results["neu"]["X"].append(msg)
            results["neu"]["y"].append(res)

    save_json(results["pos"], "assets/pos.json")
    save_json(results["neg"], "assets/neg.json")
    save_json(results["neu"], "assets/neu.json")

