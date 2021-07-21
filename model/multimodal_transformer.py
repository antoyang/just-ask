from transformers.activations import gelu
import torch.nn as nn
import numpy as np
import torch
import math
from model.language_model import Bert, AModel
import copy
from transformers.modeling_outputs import BaseModelOutput
from transformers import DistilBertConfig


def create_sinusoidal_embeddings(n_pos, dim, out):
    with torch.no_grad():
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(config.activation)
        self.activation = gelu if config.activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output
        )  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        x,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
    ):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            if head_mask is not None:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=None,
                    output_attentions=output_attentions,
                )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_state, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Embeddings(nn.Module):
    def __init__(
        self, d_model, language_len, vision_len, dropout, sinusoidal_pos_embds
    ):
        super().__init__()
        max_position_embeddings = language_len + vision_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                out=self.position_embeddings.weight,
            )
        self.modality_embedding = nn.Embedding(2, d_model)
        self.language_len = language_len
        self.vision_len = vision_len
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(
            embeddings[:, :, 0]
        )  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)
        modality_embeddings = self.modality_embedding(
            torch.tensor(
                [0] * self.language_len + [1] * self.vision_len, dtype=torch.long
            ).to(embeddings.device)
        )
        embeddings = (
            embeddings + position_embeddings + modality_embeddings
        )  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class MMT_VideoQA(nn.Module):
    def __init__(
        self,
        feature_dim=1024,
        word_dim=768,
        N=2,
        h=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        Q=20,
        T=20,
        vocab_size=30522,
        baseline="",
        n_negs=1,
    ):
        """
        :param feature_dim: dimension of the input video features
        :param word_dim: dimension of the input question features
        :param N: number of transformer layers
        :param h: number of transformer heads
        :param d_model: dimension for the transformer and final embedding
        :param d_ff: hidden dimension in the transformer
        :param dropout: dropout rate in the transformer
        :param Q: maximum number of tokens in the question
        :param T: maximum number of video features
        :param vocab_size: size of the vocabulary for the masked language modeling head
        :param baseline: set as "qa" not to use the video
        :param n_negs: number of negatives sampled for cross-modal matching
        """
        super(MMT_VideoQA, self).__init__()
        # video modules
        self.linear_video = nn.Linear(feature_dim, d_model)
        self.norm_video = nn.LayerNorm(d_model, eps=1e-12)

        # question post bert modules
        self.linear_question = nn.Linear(word_dim, d_model)
        self.norm_question = nn.LayerNorm(d_model, eps=1e-12)

        # positional and modality encoding
        self.position = Embeddings(d_model, Q, T, dropout, True)

        # video and question fusion modules
        self.config = DistilBertConfig.from_pretrained(
            "distilbert-base-uncased",
            n_layers=N,
            dim=d_model,
            dropout=dropout,
            hidden_dim=d_ff,
            attention_dropout=dropout,
            n_heads=h,
        )
        self.mmt = Transformer(self.config)
        self.vqproj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, d_model))

        # parameters
        self.baseline = baseline
        self.Q = Q
        self.T = T
        self.n_negs = n_negs

        # masked language modeling head
        self.vocab_transform = nn.Linear(d_model, d_model)
        self.vocab_norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-12)
        self.vocab_projector = nn.Linear(d_model, vocab_size)
        self.mlm_loss_fct = nn.CrossEntropyLoss()

        # cross-modal matching head
        self.crossmodal_matching = nn.Linear(d_model, 1)
        self.cm_loss_fct = nn.BCELoss()

        # weight initialization
        self.apply(self._init_weights)
        self.answer_embeddings = None

        # pretrained DistilBERT language model
        self.bert = Bert()

        # answer modules
        self.amodel = AModel(out_dim=d_model, sentence_dim=2048)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _compute_answer_embedding(self, a2v):
        self.answer_embeddings = self.get_answer_embedding(a2v)

    def get_answer_embedding(self, answer):
        answer = self.amodel(answer)
        return answer

    def get_video_embedding(self, video):
        video = self.linear_video(video)
        video = gelu(video)
        video = self.norm_video(video)
        return video

    def get_question_embedding(self, question):
        question = self.linear_question(question)
        question = gelu(question)
        question = self.norm_question(question)
        return question

    def forward(
        self,
        video,
        question=None,
        labels=None,
        answer=None,
        video_mask=None,
        text_mask=None,
        mode="vqa",
    ):
        """
        :param video: [bs, T, feature_dim]
        :param question: [bs, Q]
        :param labels: [bs, Q] used for masked language modeling
        :param answer: [batch_size, amax_words, 300] used for contrastive loss training, otherwise precomputed at the vocabulary level
        :param video_mask: [bs, T]
        :param text_mask: [bs, Q]
        """
        if mode == "vqa":
            question = self.bert(question)
            if question.shape[1] < self.Q:
                question = torch.cat(
                    [
                        question,
                        torch.zeros(
                            question.shape[0],
                            self.Q - question.shape[1],
                            question.shape[2],
                        ).cuda(),
                    ],
                    1,
                )
                text_mask = torch.cat(
                    [
                        text_mask,
                        torch.zeros(
                            text_mask.shape[0], self.Q - text_mask.shape[1]
                        ).cuda(),
                    ],
                    1,
                )
            if self.baseline == "qa":
                question_proj = self.get_question_embedding(question)
                vq_cat = torch.cat(
                    [
                        question_proj,
                        torch.zeros(
                            question_proj.size(0), self.T, question_proj.size(-1)
                        ).cuda(),
                    ],
                    dim=1,
                )
                vq = self.position(vq_cat)
                mask = torch.cat(
                    [text_mask, torch.zeros(question_proj.size(0), self.T).cuda()],
                    dim=1,
                )
                attended_vq = self.mmt(x=vq, attn_mask=mask)[0]
                fusion_proj = self.vqproj(attended_vq[:, 0, :])
            else:
                video_proj = self.get_video_embedding(video)
                question_proj = self.get_question_embedding(question)
                vq_cat = torch.cat([question_proj, video_proj], dim=1)
                mask = torch.cat([text_mask, video_mask], dim=1)
                vq = self.position(vq_cat)
                attended_vq = self.mmt(x=vq, attn_mask=mask)[0]
                fusion_proj = self.vqproj(attended_vq[:, 0, :])
            answer_proj = (
                self.get_answer_embedding(answer)
                if answer is not None
                else self.answer_embeddings
            )
            if question is not None and answer_proj.device != question.device:
                answer_proj = answer_proj.to(question.device)
            if answer is not None:
                return fusion_proj, answer_proj
            else:
                return fusion_proj @ answer_proj.t()

        elif mode == "mlm":
            if text_mask.shape[1] < self.Q:
                text_mask = torch.cat(
                    [
                        text_mask,
                        torch.zeros(
                            text_mask.shape[0], self.Q - text_mask.shape[1]
                        ).cuda(),
                    ],
                    1,
                )
                labels = torch.cat(
                    [
                        labels,
                        -100
                        * torch.ones(labels.shape[0], self.Q - labels.shape[1])
                        .long()
                        .cuda(),
                    ],
                    1,
                )
            mask = torch.cat([text_mask, video_mask], dim=1)
            video_proj = self.get_video_embedding(video)
            text = self.bert(question)
            if text.shape[1] < self.Q:
                text = torch.cat(
                    [
                        text,
                        torch.zeros(
                            text.shape[0], self.Q - text.shape[1], text.shape[2]
                        ).cuda(),
                    ],
                    1,
                )
            text_proj = self.get_question_embedding(text)
            vq_cat = torch.cat([text_proj, video_proj], dim=1)
            vq = self.position(vq_cat)
            attended_vq = self.mmt(x=vq, attn_mask=mask)[0]
            prediction_logits = self.vocab_transform(attended_vq[:, : self.Q, :])
            prediction_logits = gelu(prediction_logits)
            prediction_logits = self.vocab_norm(prediction_logits)
            prediction_logits = self.vocab_projector(prediction_logits)
            mlm_loss = self.mlm_loss_fct(
                prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1)
            )
            return mlm_loss

        elif mode == "cm":
            batch_size = len(video)
            video_proj = self.get_video_embedding(video)
            text = self.bert(question)
            if text.shape[1] < self.Q:
                text = torch.cat(
                    [
                        text,
                        torch.zeros(
                            text.shape[0], self.Q - text.shape[1], text.shape[2]
                        ).cuda(),
                    ],
                    1,
                )
                text_mask = torch.cat(
                    [
                        text_mask,
                        torch.zeros(
                            text_mask.shape[0], self.Q - text_mask.shape[1]
                        ).cuda(),
                    ],
                    1,
                )
            text_proj = self.get_question_embedding(text)

            positives_vt = torch.cat([text_proj, video_proj], dim=1)
            positives_vtembd = self.position(positives_vt)
            mask = torch.cat([text_mask, video_mask], dim=1)
            positives_attended_vt = self.mmt(x=positives_vtembd, attn_mask=mask)[0]
            positives_scores = torch.sigmoid(
                self.crossmodal_matching(positives_attended_vt[:, 0, :])
            ).squeeze()
            positives_loss = self.cm_loss_fct(
                positives_scores, torch.ones(batch_size).cuda()
            )

            rd_idx = (
                np.random.choice(
                    np.arange(1, batch_size), size=batch_size, replace=True
                )
                + np.arange(batch_size)
            ) % batch_size
            video_proj_negs = video_proj[rd_idx]
            video_negs_mask = video_mask[rd_idx]
            video_negatives_vt = torch.cat([text_proj, video_proj_negs], dim=1)
            video_negatives_vtembd = self.position(video_negatives_vt)
            mask_vnegs = torch.cat([text_mask, video_negs_mask], dim=1)
            video_negatives_attended_vt = self.mmt(
                x=video_negatives_vtembd, attn_mask=mask_vnegs
            )[0]
            video_negatives_scores = torch.sigmoid(
                self.crossmodal_matching(video_negatives_attended_vt[:, 0, :])
            ).squeeze()
            video_negatives_loss = self.cm_loss_fct(
                video_negatives_scores, torch.zeros(batch_size).cuda()
            )

            rd_idx_txt = (
                np.random.choice(
                    np.arange(1, batch_size), size=batch_size, replace=True
                )
                + np.arange(batch_size)
            ) % batch_size
            text_proj_negs = text_proj[rd_idx_txt]
            text_negs_mask = text_mask[rd_idx_txt]
            text_negatives_vt = torch.cat([text_proj_negs, video_proj], dim=1)
            text_negatives_vtembd = self.position(text_negatives_vt)
            mask_tnegs = torch.cat([text_negs_mask, video_mask], dim=1)
            text_negatives_attended_vt = self.mmt(
                x=text_negatives_vtembd, attn_mask=mask_tnegs
            )[0]
            text_negatives_scores = torch.sigmoid(
                self.crossmodal_matching(text_negatives_attended_vt[:, 0, :])
            ).squeeze()
            text_negatives_loss = self.cm_loss_fct(
                text_negatives_scores, torch.zeros(batch_size).cuda()
            )

            cm_loss = positives_loss + video_negatives_loss + text_negatives_loss
            return cm_loss

        elif mode == "retrieval":
            text = self.bert(question)
            if text.shape[1] < self.Q:
                text = torch.cat(
                    [
                        text,
                        torch.zeros(
                            text.shape[0], self.Q - text.shape[1], text.shape[2]
                        ).cuda(),
                    ],
                    1,
                )
                text_mask = torch.cat(
                    [
                        text_mask,
                        torch.zeros(
                            text_mask.shape[0], self.Q - text_mask.shape[1]
                        ).cuda(),
                    ],
                    1,
                )
            text_proj = self.get_question_embedding(text)
            text_proj_rep = text_proj.repeat(len(video), 1, 1)
            text_mask_rep = text_mask.repeat(len(video), 1)
            video_proj = self.get_video_embedding(video)
            vt = torch.cat([text_proj_rep, video_proj], dim=1)
            mask = torch.cat([text_mask_rep, video_mask], dim=1)
            attended_vt = self.mmt(x=vt, attn_mask=mask)[0]
            scores = torch.sigmoid(
                self.crossmodal_matching(attended_vt[:, 0, :])
            ).squeeze()
            return scores

        elif mode == "vqacm":
            text = self.bert(question.squeeze())
            text_mask = text_mask.squeeze()
            if text.shape[1] < self.Q:
                text = torch.cat(
                    [
                        text,
                        torch.zeros(
                            text.shape[0], self.Q - text.shape[1], text.shape[2]
                        ).cuda(),
                    ],
                    1,
                )
                text_mask = torch.cat(
                    [
                        text_mask,
                        torch.zeros(
                            text_mask.shape[0], self.Q - text_mask.shape[1]
                        ).cuda(),
                    ],
                    1,
                )
            text_proj = self.get_question_embedding(text)
            video_proj = self.get_video_embedding(video)
            video_proj_rep = video_proj.repeat(len(text), 1, 1)
            video_mask_rep = video_mask.repeat(len(text), 1)
            vt = torch.cat([text_proj, video_proj_rep], dim=1)
            mask = torch.cat([text_mask, video_mask_rep], dim=1)
            attended_vt = self.mmt(x=vt, attn_mask=mask)[0]
            scores = torch.sigmoid(
                self.crossmodal_matching(attended_vt[:, 0, :])
            ).squeeze()
            return scores
