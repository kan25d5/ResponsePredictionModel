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


def beam_search(model, source: Tensor):
    def get_prob(model, ys, memory) -> Tensor:
        """Decoderの順伝播"""
        tgt_mask, _ = model._create_tgt_mask(ys)
        ys_emb_pe = model.pe(model.tgt_tok_emb(ys))
        out = model.decoder(ys_emb_pe, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generater(out[:, -1])
        return prob

    def truncate_after_EOS(node_ys: Tensor):
        new_tensor = []
        for item in node_ys:
            item = item[0].item()
            new_tensor.append([item])
            if item == 2:
                break
        return torch.LongTensor(new_tensor)

    # top-beam_sizeのノードを保持
    node_list = []

    # Encoderの順伝播
    src_mask, _ = model._create_src_mask(source)
    src_emb_pe = model.pe(model.src_tok_emb(source))
    memory = model.encoder(src_emb_pe, src_mask)
    ys = torch.ones(1, 1, device=model.device)

    # １単語目の推論
    prob = get_prob(model, ys, memory)
    top_values, top_indices = prob.topk(model.beam_size, dim=-1)
    for node_i, (value, indice) in enumerate(zip(top_values[0], top_indices[0])):
        prob, word = value.item(), indice.item()
        node = BeamNode(word, ys, source, prob, node_i)
        node_list.append(node)

    # ２単語目以降の推論
    aleady_reached_eos = []
    for word_i in range(1, model.maxlen - 1, 1):
        for node_index in range(model.beam_size):
            prev_node = node_list[node_index]
            ys = prev_node.ys

            if node_index in aleady_reached_eos:
                continue

            top_node: BeamNode
            top_node_score = -9999
            prob = get_prob(model, ys, memory)
            top_values, top_indices = prob.topk(model.beam_size, dim=-1)
            for value, indice in zip(top_values[0], top_indices[0]):
                prob, word = value.item(), indice.item()
                node = BeamNode(word, ys, source, prob, node_index, word_i, prev_node)
                score = node.eval()
                if score > top_node_score:
                    top_node = node
                    top_node_score = score

            node_list[node_index] = top_node

            if top_node.next_word == 2:
                aleady_reached_eos.append(node_index)

    # 最大スコアのノードだけを返す
    top_node: BeamNode
    top_node_score = -9999
    for node in node_list:
        score = node.eval()
        if score > top_node_score:
            top_node = node
            top_node_score = score

    ys = truncate_after_EOS(top_node.ys)
    return ys


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
