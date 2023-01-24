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


class DPRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_encoder = DPREncoder.from_pretrained(config.model.name_or_path)
        self.ctx_encoder = DPREncoder.from_pretrained(config.model.name_or_path)

    def forward(self, inputs):
        query_tensor = inputs["question_tensor"]  # [batch_size, embed_dim]
        ctx_tensor = inputs["answer_tensor"]  # [batch_size, embed_dim]

        query_output = self.query_encoder(**query_tensor)
        ctx_output = self.ctx_encoder(**ctx_tensor)

        sim_scores = torch.matmul(query_output, ctx_output.T)  # [batch_size, batch_size]

        return sim_scores
