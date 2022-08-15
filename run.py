from torch.utils.data import DataLoader
from dataloader.twitter_dataset import TwitterDataset
from vocab.twitter_vocab import TwitterVocab
from utilities.training_functions import get_corpus

X, y = get_corpus()
vocab = TwitterVocab()
vocab.load_char2id_pkl()
dataset = TwitterDataset(X, y, vocab)

for i in range(10):
    print(dataset[i])

dataloader = DataLoader(dataset, batch_size=10, collate_fn=dataset.collate_fn)
for x, y in dataloader:
    print(x)
    print(y)
