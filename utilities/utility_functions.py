import json


def load_json(filepath: str):
    with open(filepath) as f:
        json_ = json.load(f)
    return json_


def save_json(content, filepath: str):
    with open(filepath, "w") as f:
        f.write(json.dumps(content, ensure_ascii=False, indent=4))
