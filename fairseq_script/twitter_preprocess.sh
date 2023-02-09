echo -n neg/pos/neu:
read corpus_type
if [$corpus_type = ""]; then
    echo "exit"
else
    fairseq-preprocess \
     --trainpref ./assets/corpus/faitseq/$corpus_type/raw/train \
     --validpref ./assets/corpus/faitseq/$corpus_type/raw/valid \
     --testpref ./assets/corpus/faitseq/$corpus_type/raw/test \
     --source-lang src \
     --target-lang dst \
     --destdir ./assets/corpus/faitseq/$corpus_type/ \
     --tokenizer space \
     --nwordstgt 31998 \
     --nwordssrc 31998 \
     --srcdict ./assets/japanese-dialog-transformers/data/dicts/sp_oall_32k.txt \
     --tgtdict ./assets/japanese-dialog-transformers/data/dicts/sp_oall_32k.txt 
fi