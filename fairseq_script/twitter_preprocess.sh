echo "neg/pos/neu : "
read corpus
fairseq-preprocess \
  --trainpref ./assets/corpus/faitseq/raw/$corpus/train \
  --validpref ./assets/corpus/faitseq/raw/$corpus/valid \
  --testpref ./assets/corpus/faitseq/raw/$corpus/test \
  --source-lang src \
  --target-lang dst \
  --destdir ./assets/corpus/faitseq/bin/$corpus/ \
  --tokenizer space \
  --nwordstgt 31998 \
  --nwordssrc 31998 \
  --srcdict ./assets/japanese-dialog-transformers/data/dicts/sp_oall_32k.txt \
  --tgtdict ./assets/japanese-dialog-transformers/data/dicts/sp_oall_32k.txt 