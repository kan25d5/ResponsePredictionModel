import numpy as np
import torch
from torch import Tensor


class BeamNode(object):
    def __init__(
        self,
        next_word: int,
        ys: Tensor,
        source: Tensor,
        prob: float,
        node_index: int,
        word_index=0,
        prev_node=None,
        alpha=0.6,
    ) -> None:
        self.next_word = next_word
        self.prob = prob
        self.node_index = node_index
        self.word_index = word_index
        self.alpha = alpha
        self.prev_node = prev_node

        self.ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(source.data).fill_(self.next_word)], dim=0
        )

    def _get_current_node_score(self, node):
        if node.word_index > 0:
            log_likelihood = np.log(node.prob + 1e-5)
            log_likelihood /= np.power((1 + 6) / 6, self.alpha)
        elif node.word_index == 0:
            log_likelihood = np.log(node.prob + 1e-5)
        else:
            raise ValueError("nodeのindexが不正．{}".format(node))
        return log_likelihood

    def eval(self):
        node = self
        score = self._get_current_node_score(node)
        while True:
            node = node.prev_node
            if node is None:
                return score
            score += self._get_current_node_score(node)


def _get_prob(model, ys, memory) -> Tensor:
    """単語の出現確率を取得する"""
    tgt_mask, _ = model._create_tgt_mask(ys)
    ys_emb_pe = model.pe(model.tgt_tok_emb(ys))
    out = model.decoder(ys_emb_pe, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generater(out[:, -1])
    return prob


def beam_search(model, source: Tensor):
    # REVIEW: めちゃくちゃ遅いコード
    #         見直しの必要あり

    # Beam_size分だけ保持するBeamNodeリスト
    node_list = []

    # ステップ１
    # Encoderによる順伝播
    #   expected shape of ys : (B, S)
    #   expected shape prob : (B, S, V)
    src_mask, _ = model._create_src_mask(source)
    src_emb_pe = model.pe(model.src_tok_emb(source))
    memory = model.encoder(src_emb_pe, src_mask)
    ys = torch.ones(1, 1, device=model.device)

    # ステップ２
    # 最初の単語を最尤推定する
    # 生起確率上位beam_size分だけnode_listに保持する
    prob = _get_prob(model, ys, memory)
    next_values, next_words = prob.topk(model.beam_size, dim=1)

    for value, word in zip(next_values[0], next_words[0]):
        value = value.item()
        word = word.item()

        node = BeamNode(word, ys, source, value, 0, 0)
        node_list.append(node)

    # ステップ４
    # ２単語目以降の単語をビームサーチで探索する．
    # 各ノードからbeam_size分の次単語を予測し，最もスコアが高いものだけをノードとして保持（枝切り）
    for word_i in range(model.maxlen - 2):
        for i in range(model.beam_size):
            prev_node = node_list[i]
            ys = prev_node.ys

            top_node: BeamNode
            top_node_score = 0.0
            prob = _get_prob(model, ys, memory)
            next_values, next_words = prob.topk(model.beam_size, dim=1)

            for value, word in zip(next_values[0], next_words[0]):
                value = value.item()
                word = word.item()
                node = BeamNode(word, ys, source, value, i, word_i, prev_node)
                score = node.eval()
                if score > top_node_score:
                    top_node_score = score
                    top_node = node

            node_list[i] = top_node

    # ステップ５
    # 最終的に最もスコアが高いノードが保持する出力ベクトルを返す
    top_node: BeamNode
    top_score = 0
    for node in node_list:
        if node.eval() > top_score:
            top_node = node

    return node.ys


def greedy_search(model, source: Tensor):
    src_mask, _ = model._create_src_mask(source)
    src_emb_pe = model.pe(model.src_tok_emb(source))
    memory = model.encoder(src_emb_pe, src_mask)

    ys = torch.ones(1, 1, device=model.device)
    for i in range(model.maxlen - 1):
        tgt_mask, _ = model._create_tgt_mask(ys)
        ys_emb_pe = model.pe(model.tgt_tok_emb(ys))
        out = model.decoder(ys_emb_pe, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generater(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(source.data).fill_(next_word)],
            dim=0,
        )

        if next_word == 2:
            break

    return ys
