from utilities.make_sentimet_corpus import main
from utilities.training_functions import get_corpus

# main()


def display_corpus(sentimet_type: str):
    messages, responses = get_corpus(sentimet_type)
    for idx, (msg, res) in enumerate(zip(messages, responses)):
        print(msg)
        print(f"\t->{res}")
        if idx > 5:
            break
    print("-" * 50)


for label in ["pos", "neg", "neu"]:
    print("label : " + label)
    display_corpus(label)
