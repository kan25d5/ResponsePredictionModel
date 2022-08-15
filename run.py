from tqdm import tqdm
from fugashi import Tagger
from utilities.functions import load_json
from utilities.twitter_vocab import TwitterVocab


def main():
    vocab = TwitterVocab()
    parse = Tagger("-Owakati").parse
    corpus = load_json("assets/normal.json")

    X, y = [], []
    for x_txt, y_txt in tqdm(zip(corpus["X"], corpus["y"]), total=len(corpus["X"])):
        X.append(parse(x_txt))
        y.append(parse(y_txt))

    # X = [parse(text) for text in corpus["X"]]
    # y = [parse(text) for text in corpus["y"]]

    # vocab.fit(X, y)
    # vocab.save_char2id("assets/char2id.model")

    # vocab = TwitterVocab()
    vocab.load_char2id("assets/char2id.model")

    # print(len(vocab.vocab_x.char2id))
    # print(len(vocab.vocab_y.char2id))
    # print(len(vocab.vocab_x.id2char))
    # print(len(vocab.vocab_y.id2char))

    print([txt.replace(" ", "") for txt in X[0:5]])

    X = [txt for txt in X[0:5]]
    y = [txt for txt in y[0:5]]
    print(vocab.transform(X, y))


if __name__ == "__main__":
    main()
