echo "neg/pos/neu : "
read corpus
fairseq-interactive ./assets/corpus/faitseq/bin/$corpus \
 --path ./assets/fairseq_result/$corpus/checkpoint_best.pt \
 --beam 10 \
 --seed 0 \
 --min-len 10 \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model ./assets/japanese-dialog-transformers/data/dicts/sp_oall_32k.model \
 --no-repeat-ngram-size 3 \
 --nbest 10 \
 --sampling \
 --sampling-topp 0.9 \
 --temperature 1.0