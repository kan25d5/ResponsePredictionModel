echo "neg/pos/neu : "
read corpus
fairseq-train ./assets/corpus/faitseq/bin/$corpus/ \
  --arch transformer \
  --finetune-from-model ./assets/fairseq_result/neu/checkpoint_best.pt \
  --save-dir ./assets/fairseq_result/$corpus/ \
  --bpe sentencepiece \
  --sentencepiece-model ./assets/japanese-dialog-transformers/data/dicts/sp_oall_32k.model \
  --encoder-embed-dim 1920 --decoder-embed-dim 1920 \
  --encoder-attention-heads 32 --decoder-attention-heads 32 \
  --encoder-ffn-embed-dim 7680 --decoder-ffn-embed-dim 7680 \
  --encoder-layers 2 --decoder-layers 24 \
  --encoder-normalize-before --decoder-normalize-before \
  --criterion cross_entropy \
  --batch-size 20 \
  --save-interval 2 \
  --lr 0.000001 \
  --max-epoch 30 \
  --optimizer adafactor
