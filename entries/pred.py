import MeCab
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences


def _get_vocab(args):
    from vocab.twitter_vocab import TwitterVocab
    from utilities.constant import CHAR2ID_FILEPATH

    vocab = TwitterVocab()
    sentiment_type = args.sentiment_type
    maxlen = args.maxlen
    make_vocab = args.make_vocab
    data_size = args.data_size
    vocab_size = args.vocab_size

    if not make_vocab:
        vocab.load_char2id_pkl(CHAR2ID_FILEPATH)
    else:
        from utilities.training_functions import get_corpus

        X, y = get_corpus(sentiment_type=sentiment_type, maxlen=maxlen)
        X, y = X[: int(len(X) * data_size)], y[: int(len(y) * data_size)]

        vocab.fit(X, y, is_wakati=True)
        vocab.reduce_vocabulary(vocab_size)

    return vocab


def _get_model(args):
    from models.seq2seq_transform import Seq2SeqTransformer

    vocab_size = args.vocab_size
    maxlen = args.maxlen

    input_dim = vocab_size
    output_dim = vocab_size

    model = Seq2SeqTransformer(input_dim, output_dim, maxlen=maxlen)
    # model.load_from_checkpoint("assets/persona-v1.ckpt")
    model.load_state_dict(torch.load("assets/persona-v1.ckpt")["state_dict"])
    return model


def pred(args):
    maxlen = args.maxlen

    transform = MeCab.Tagger("-Owakati").parse
    vocab = _get_vocab(args)
    model = _get_model(args)

    while True:
        src = input("input : ")
        if src == "":
            break

        src = transform(src)
        src = vocab.vocab_X.encode(src, is_wakati=True)
        src = pad_sequences([src], maxlen=maxlen, padding="post")
        src = torch.LongTensor(src).t()

        pred = model(src)
        target = [tgt[0].item() for tgt in pred]
        target = vocab.vocab_y.decode(target)

        print("output : {}".format(target))
        print("=" * 20)
