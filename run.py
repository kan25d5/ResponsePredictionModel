from vocab.twitter_vocab import TwitterVocab

vocab = TwitterVocab()
vocab.load_char2id_pkl()


from utilities.training_functions import get_corpus, get_dataset, get_dataloader

X, y = get_corpus()
all_dataset = get_dataset(X, y, vocab)
all_dataloader = get_dataloader(all_dataset)
train_dataloader = all_dataloader[0]

for x, y in train_dataloader:
    print(x)
    print(y)
    break
