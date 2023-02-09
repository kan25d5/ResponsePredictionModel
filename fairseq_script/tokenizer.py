import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("assets/japanese-dialog-transformers/data/dicts/sp_oall_32k.model")


def tokenizer(raw_text: str):
    return " ".join(sp.EncodeAsPieces(raw_text))


text = "はあ。わたしなで肩がコンプレックスでさ。"
print("origin : " + text)
print("\t->" + tokenizer(text))
