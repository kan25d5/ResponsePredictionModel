echo -n neg/pos/neu:
read corpus_type
fairseq-train ./assets/corpus/faitseq/$corpus_type/ \
 --arch transformer \
 --finetune-from-model ./assets/japanese-dialog-transformers/checkpoints/japanese-dialog-transformer-1.6B.pt \
 --save-dir ./assets/fairseq_result/$corpus_type/ \
 --bpe sentencepiece \
 --sentencepiece-model ./assets/japanese-dialog-transformers/data/dicts/sp_oall_32k.model \
 --encoder-embed-dim 1920 --decoder-embed-dim 1920 \
 --encoder-attention-heads 32 --decoder-attention-heads 32 \
 --encoder-ffn-embed-dim 7680 --decoder-ffn-embed-dim 7680 \
 --encoder-layers 2 --decoder-layers 24 \
 --encoder-normalize-before --decoder-normalize-before \
 --criterion cross_entropy \
 --batch-size 10 \
 --save-interval 5 \
 --lr 0.000001 \
 --max-epoch 20 \
 --optimizer adafactor
