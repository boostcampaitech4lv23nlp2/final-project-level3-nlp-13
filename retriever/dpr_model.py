import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel


class DPREncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertModel(config)
        self.init_weights

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs["pooler_output"]
