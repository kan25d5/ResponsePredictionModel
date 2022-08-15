from vocab.twitter_vocab import TwitterVocab
from utilities.training_functions import get_corpus


def make_char2id():
    X, y = get_corpus()
    vocab = TwitterVocab()

    vocab.fit(X, y, is_wakati=False)
    vocab.save_char2id_pkl()


def check_vocab():
    X, y = get_corpus()
    vocab = TwitterVocab()
    vocab.load_char2id_pkl()

    batch_x = vocab.vocab_X.transform(X[0:3], is_wakati=False)
    batch_x_converted = vocab.vocab_X.convert_ids_to_str(batch_x)
    print(batch_x_converted)


# make_char2id()
check_vocab()
