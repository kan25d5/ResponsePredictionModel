def get_xy(sentimet_type: str):
    from utilities.functions import load_json
    from fugashi import Tagger

    parse = Tagger("-Owakati").parse
    filepath = f"assets/{sentimet_type}.json"

    corpus = load_json(filepath)
    X = [parse(txt) for txt in corpus["X"]]
    y = [parse(txt) for txt in corpus["y"]]

    return X, y

