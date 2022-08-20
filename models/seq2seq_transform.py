import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import Tuple
from torch import Tensor
from layers.seq2seq_transformer_layers import PositionalEncoding, TokenEmbedding
from vocab.twitter_vocab import TwitterVocab


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_layers=6,
        emb_size=512,
        maxlen=140,
        padding_idx=0,
        eos_idx=2,
        learning_ratio=0.0001,
    ) -> None:
        super().__init__()

        # フィールド値の定義
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_size = emb_size
        self.d_model = emb_size
        self.nhead = self.d_model // 64
        self.maxlen = maxlen
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        self.learning_ratio = learning_ratio

        # レイヤーの定義
        self.src_tok_emb = TokenEmbedding(
            src_vocab_size, self.emb_size, padding_idx=self.padding_idx
        )
        self.tgt_tok_emb = TokenEmbedding(
            tgt_vocab_size, self.emb_size, padding_idx=self.padding_idx
        )
        self.pe = PositionalEncoding(self.d_model, max_len=self.maxlen, device=self.device)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(self.d_model, self.nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.generater = nn.Linear(self.d_model, tgt_vocab_size)

        # 損失関数の定義
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        # 評価手法
        self.test_acc = Accuracy()

    def _training(self, source: Tensor, target: Tensor):
        tgt_input = target[:-1, :]
        src_emb_pe = self.pe(self.src_tok_emb(source))
        tgt_emb_pe = self.pe(self.tgt_tok_emb(tgt_input))
        src_mask, src_padding_mask = self._create_src_mask(source)
        tgt_mask, tgt_padding_mask = self._create_tgt_mask(tgt_input)

        memory = self.encoder(src_emb_pe, src_mask, src_padding_mask)
        out = self.decoder(tgt_emb_pe, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)
        out = self.generater(out)

        return out

    def _predict(self, source: Tensor):
        src_mask, _ = self._create_src_mask(source)
        src_emb_pe = self.pe(self.src_tok_emb(source))
        memory = self.encoder(src_emb_pe, src_mask)

        ys = torch.ones(1, 1, device=self.device)
        for i in range(self.maxlen - 1):
            tgt_mask, _ = self._create_tgt_mask(ys)
            ys_emb_pe = self.pe(self.tgt_tok_emb(ys))
            out = self.decoder(ys_emb_pe, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generater(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            next_tensor = torch.ones(1, 1, device=self.device)
            next_tensor = next_tensor.type_as(source.data).fill_(next_word)
            ys = torch.cat([ys, next_tensor], dim=0)

            if next_word == 2:
                break

        return ys

    def forward(self, source: Tensor, target: Tensor = None):
        if target is None:
            return self._predict(source)
        else:
            return self._training(source, target)

    def _create_src_mask(self, src: Tensor):
        src_size = src.shape[0]
        src_mask = torch.zeros((src_size, src_size), device=self.device).type(torch.bool)
        src_padding_mask = (src == self.padding_idx).transpose(0, 1)
        return src_mask.to(self.device), src_padding_mask.to(self.device)

    def _create_tgt_mask(self, tgt: Tensor):
        tgt_size = tgt.shape[0]
        tgt_mask = self._generate_square_subsequent_mask(tgt_size)
        tgt_padding_mask = (tgt == self.padding_idx).transpose(0, 1).to(self.device)
        return tgt_mask.to(self.device), tgt_padding_mask.to(self.device)

    def _generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill_(mask == 0, float("-inf"))
        mask = mask.masked_fill_(mask == 1, float(0.0))
        return mask.to(self.device)

    def compute_loss(self, preds: Tensor, target: Tensor):
        preds = preds.reshape(-1, preds.shape[-1])
        target = target.reshape(-1)
        loss = self.criterion(preds, target)
        return loss

    def compute_acc(self, preds: Tensor, target: Tensor, acc):
        preds = preds.reshape(-1, preds.shape[-1])
        target = target.reshape(-1)
        return acc(preds, target)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self.train()

        x, t = batch
        tgt_out = t[1:, :]
        preds = self.forward(x, t)

        loss = self.compute_loss(preds, tgt_out)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "source": x.to("cpu"),
            "target": t.to("cpu"),
            "preds": preds.to("cpu"),
        }

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, t = batch
        tgt_out = t[1:, :]
        preds = self.forward(x, t)
        loss = self.compute_loss(preds, tgt_out)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "source": x.to("cpu"),
            "target": t.to("cpu"),
            "preds": preds.to("cpu"),
        }

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, t = batch
        tgt_out = t[1:, :]
        preds = self.forward(x, t)

        loss = self.compute_loss(preds, tgt_out)
        self.log("test_loss", value=loss)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_ratio, betas=(0.9, 0.98), eps=1e-9)

