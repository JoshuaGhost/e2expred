from transformers import BertModel, BertTokenizer
from typing import Any, List

import torch
import torch.nn as nn
from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_bert import BERT_INPUTS_DOCSTRING

from latent_rationale.mtl_e2e.utils import PaddedSequence
from latent_rationale.nn.kuma_gate import KumaGate
from latent_rationale.common.latent import IndependentSelector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["BertDataPreprocessor", "KumaSelectorLayer", "BertClassifier"]


class BertClassificationHead(nn.Module):
    def __init__(self, input_size, cls_head_params, num_labels):
        super(BertClassificationHead, self).__init__()
        self.nn = nn.Sequential(
            nn.Dropout(cls_head_params['dropout']),
            nn.Linear(input_size, cls_head_params['dim_hidden'], bias=True),
            nn.Tanh(),
            nn.Linear(cls_head_params['dim_hidden'], num_labels, bias=True),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, **kwargs: Any):
        return self.nn(x)


class BertDataPreprocessor(nn.Module):
    def __init__(self, cfg):
        super(BertDataPreprocessor, self).__init__()
        pass

    def forward(self, query, document_batch, *args: Any, **kwargs: Any):
        assert len(query) == len(document_batch)
        # print(next(self.cls_head.parameters()).device)
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id]).to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id]).to(device=document_batch[0].device)
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[:(self.max_length - len(q) - 2)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d]))
            position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))  # tokens
            # positions are counted from 1, the two 0s are for [cls] and [sep], [pad]s are also nominated as pos 0
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id,
                                            device=target_device)
        positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=target_device)
        attention_masks = bert_input.mask(on=1., off=0., device=target_device)
        return bert_input, attention_masks, positions


# class BertMTLLayer(nn.Module):
#     def __init__(self,
#                  mtl_params,
#                  tokenizer: BertTokenizer):
#         super(BertMTLLayer, self).__init__()
#         bert_dir = mtl_params.bert_dir
#         use_half_precision = bool(mtl_params.use_half_precision)
#         bare_bert = BertModel.from_pretrained(bert_dir)
#         if use_half_precision:
#             import apex
#             bare_bert = bare_bert.half()
#         self.bare_bert = bare_bert
#         self.num_label = mtl_params.num_label
#         self.cls_head = BertClassificationHead(self.bare_bert.config.hidden_size,
#                                                mtl_params.cls_head,
#                                                self.num_label)
#         self.pad_token_id = tokenizer.pad_token_id
#         self.cls_token_id = tokenizer.cls_token_id
#         self.sep_token_id = tokenizer.sep_token_id
#         self.max_length = mtl_params.max_length
#
#     def forward(self,
#                 bert_input: PaddedSequence,
#                 attention_mask: torch.Tensor,
#                 positions: PaddedSequence):
#         exp_output, cls_output_hidden = self.bare_bert(bert_input.data,
#                                                        attention_mask=attention_mask,
#                                                        position_id=positions.data)
#         cls_output = self.cls_head(cls_output_hidden)
#
#         assert torch.all(cls_output == cls_output)
#         assert torch.all(exp_output == exp_output)
#         return cls_output, exp_output


class KumaSelectorLayer(IndependentSelector):
    def __init__(self,
                 selector_params,
                 repr_size: int):
        super(KumaSelectorLayer, self).__init__()
        dropout = selector_params['dropout']
        distribution = selector_params['dist']
        self.selector_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=repr_size, out_features=repr_size),
            # nn.Tanh(),
            # nn.Linear(in_features=repr_size, out_features=repr_size),
            # nn.Tanh()
        )
        if distribution == "kuma":
            self.z_layer = KumaGate(repr_size)
        else:
            raise ValueError("unknown distribution")
        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)
        self.report_params()

    def forward(self, x, mask, **kwargs):

        # encode sentence
        # lengths = mask.sum(1)

        h = self.selector_head(x)

        z_dist = self.z_layer(h)

        # we sample once since the state was already repeated num_samples
        if self.training:
            if hasattr(z_dist, "rsample"):
                z = z_dist.rsample()  # use rsample() if it's there
            else:
                z = z_dist.sample()  # [B, M, 1]
        else:
            # deterministic strategy
            p0 = z_dist.pdf(h.new_zeros(()))
            p1 = z_dist.pdf(h.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            z = torch.where(p0 > p1, h.new_zeros([1]), h.new_ones([1]))
            z = torch.where((pc > p0) & (pc > p1), z_dist.mean(), z)  # [B, M, 1]

        # mask invalid positions
        z = z.squeeze(-1)
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z  # [B, T]
        self.z_dists = [z_dist]

        return z


class BertModelWithEmbeddingMask(BertModel):
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            embedding_masks=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # if embedding_masks is not None:
        #     print(f"embedding_output_head = {embedding_output[0,:8,:8]}")
        #     print(f"embedding_masks_head = {embedding_masks[0]}")
        if embedding_masks is not None:
            embedding_output *= embedding_masks.unsqueeze(-1)
        # if embedding_masks is not None:
        #     print(f"embedding_output_head = {embedding_output[0,:8,:8]}")
        #     print(f"embedding_masks_head = {embedding_masks[0]}")

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertClassifier(nn.Module):
    def __init__(self,
                 classifier_params,
                 tokenizer: BertTokenizer, ):
        super(BertClassifier, self).__init__()
        self.num_labels = classifier_params['num_labels']
        self.cls_params = classifier_params['cls_head']
        self.max_length = classifier_params['max_length']
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        bert_dir = classifier_params['bert_dir']
        bert_model = BertModelWithEmbeddingMask.from_pretrained(bert_dir)
        if bool(classifier_params['use_half_precision']):
            import apex
            bert_model = bert_model.half()
        self.bert_model = bert_model
        self.cls_head = BertClassificationHead(self.bert_model.config.hidden_size,
                                               self.cls_params,
                                               self.num_labels)
        for layer in self.cls_head.children():
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self,
                bert_inputs: torch.Tensor,
                attention_masks: torch.Tensor = None,
                embedding_masks: torch.Tensor = None,
                positions: torch.Tensor = None):
        exp_output, cls_output_hidden = self.bert_model(bert_inputs.data,
                                                        attention_mask=attention_masks,
                                                        embedding_masks=embedding_masks,
                                                        position_ids=positions)
        cls_p = self.cls_head(cls_output_hidden)

        assert torch.all(cls_p == cls_p)
        assert torch.all(exp_output == exp_output)
        return cls_p, exp_output


