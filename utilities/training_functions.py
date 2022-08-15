from typing import List
from utilities.utility_functions import load_json


def get_corpus(sentiment_type: str = "normal"):
    filepath = f"assets/{sentiment_type}.json"
    corpus = load_json(filepath)
    return corpus["X"], corpus["y"]
