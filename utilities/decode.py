import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class BeamNode(object):
    def __init__(
        self,
        next_word: int,
        ys: Tensor,
        source: Tensor,
        index=0,
        prob: float = 0,
        prev_node=None,
        alpha=0.6,
    ) -> None:
        self.index = index
        self.next_word = next_word
        self.prev_node = prev_node
        self.alpha = alpha
        self._set_ys(ys, source)

        if prob == 0:
            self.score = 0
        else:
            self._set_score(prob)

    def _set_score(self, prob):
        self.score = np.log(prob) / (((6 + self.index) / 6) ** self.alpha)

    def _set_ys(self, ys: Tensor, source: Tensor):
        new_ys = torch.ones(1, 1).type_as(source.data).fill_(self.next_word)
        self.ys = torch.cat([ys, new_ys], dim=0)

    def get_score(self):
        # TODO: スコア関数を再考する
        score = self.score
        node = self.prev_node
        while True:
            if node is None:
                break
            score += node.score
            node = node.prev_node
        return score

    def __str__(self) -> str:
        return f"next_word={self.next_word}, {self.next_word}\n \
            prob={self.prob}\n \
            prev_node={self.prev_node}\n \
            ys={self.ys}, {self.ys.size()}\n"


def get_prob(model, ys, memory) -> Tensor:
    tgt_mask, _ = model._create_tgt_mask(ys)
    ys_emb_pe = model.pe(model.tgt_tok_emb(ys))
    out = model.decoder(ys_emb_pe, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generater(out[:, -1])
    return prob


def beam_decode(model, source: Tensor):
    # REVIEW: めちゃくちゃ遅いコード
    #         見直しの必要あり

    # Beam_size分だけ保持するBeamNodeリスト
    node_list = []

    # ステップ１
    # Encoderによる順伝播
    src_mask, _ = model._create_src_mask(source)
    src_emb_pe = model.pe(model.src_tok_emb(source))
    memory = model.encoder(src_emb_pe, src_mask)
    ys = torch.ones(1, 1, device=model.device)

    # ステップ２
    # 最初の単語を最尤推定する
    prob = get_prob(model, ys, memory)
    next_values, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()
    first_node = BeamNode(next_word, ys, source)

    # ステップ３
    # ２単語目をbeam_size分予測し，その分のbeam_listを作る
    prev_node = first_node
    prob = get_prob(model, prev_node.ys, memory)
    next_values, next_words = prob.topk(model.beam_size, dim=1)

    for i in range(model.beam_size):
        ys = prev_node.ys
        value = next_values[0][i].item()
        word = next_words[0][i].item()
        node = BeamNode(word, ys, source, 1, value, prev_node)
        node_list.append(node)

    # ステップ４
    # ２単語目以降の単語をビームサーチで探索する．
    # 各ノードからbeam_size文の次単語を予測し，最もスコアが高いものだけをノードとして保持（枝切り）
    
    for word_i in range(model.maxlen - 2):
        for i in range(model.beam_size):
            prev_node = node_list[i]
            ys = prev_node.ys

            prob = get_prob(model, ys, memory)
            next_values, next_words = prob.topk(model.beam_size, dim=1)

            top_node: BeamNode
            top_score = 0
            for j in range(model.beam_size):
                value = next_values[0][j].item()
                word = next_words[0][j].item()
                node = BeamNode(word, ys, source, word_i, value, prev_node)

                if node.get_score() > top_score:
                    top_node = node

            node_list[i] = top_node

    # ステップ５
    # 最終的に最もスコアが高いノードが保持する出力ベクトルを返す
    top_node: BeamNode
    top_score = 0
    for node in node_list:
        if node.get_score() > top_score:
            top_node = node

    return node.ys


def training(model, source: Tensor, target: Tensor) -> Tensor:
    # REVIEW: BeamSearchを使ったtrainingの方法はあるのか．

    tgt_input = target[:-1, :]
    src_emb_pe = model.pe(model.src_tok_emb(source))
    tgt_emb_pe = model.pe(model.tgt_tok_emb(tgt_input))
    src_mask, src_padding_mask = model._create_src_mask(source)
    tgt_mask, tgt_padding_mask = model._create_tgt_mask(tgt_input)
    memory = model.encoder(src_emb_pe, src_mask, src_padding_mask)
    out = model.decoder(tgt_emb_pe, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)
    out = model.generater(out)
    return out


def predict(model, source: Tensor) -> Tensor:
    return beam_decode(model, source)
