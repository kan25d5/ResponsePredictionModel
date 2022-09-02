import random
from tqdm import tqdm
from MeCab import Tagger
from vocab.twitter_vocab import TwitterVocab
from utilities.utility_functions import load_json
from utilities.constant import MECAB_USER_DICT

tagger = Tagger("-Owakati " + MECAB_USER_DICT)

NORMAL_CORPUS_FILE = "assets/normal.json"


def make_corpus():
    messages = []
    responses = []

    corpus = load_json(NORMAL_CORPUS_FILE)
    for x, y in tqdm(zip(corpus["X"], corpus["y"]), total=len(corpus["X"])):
        messages.append(tagger.parse(x))
        responses.append(tagger.parse(y))

    return messages, responses


def make_vocab(X, y):
    vocab = TwitterVocab()
    vocab.fit(X, y)

    vocab.save_char2id_pkl()
    return vocab


def main():
    X, y = make_corpus()
    vocab = make_vocab(X, y)

    for i in range(20):
        print(f"vocab_X.id2char[{i}] is {vocab.vocab_X.id2char[i]}    ", end="")
        print(f"vocab_y.id2char[{i}] is {vocab.vocab_y.id2char[i]}")
    for _ in range(10):
        i = random.randint(0, 100000)
        print(f"vocab_X.id2char[{i}] is {vocab.vocab_X.id2char[i]}    ", end="")
        print(f"vocab_y.id2char[{i}] is {vocab.vocab_y.id2char[i]}")

    print(f"len(vocab.vocab_X.char2id) : {len(vocab.vocab_X.char2id)}")
    print(f"len(vocab.vocab_y.char2id) : {len(vocab.vocab_y.char2id)}")


if __name__ == "__main__":
    main()
