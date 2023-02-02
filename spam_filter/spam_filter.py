from dataclasses import dataclass, field
import argparse
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, PreTrainedTokenizerFast, BertForSequenceClassification

@dataclass
class SpamFilter:
    def __post_init__(self):
        self.spam_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        self.spam_model = BertForSequenceClassification.from_pretrained('nlpotato/spam-filtering-bert-base-10e') #훈련된 bert 모델

    def sentences_predict(self, sent):        
        self.spam_model.eval()
        tokenized_sent = self.spam_tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                add_special_tokens=True,
                max_length=128
        )

        with torch.no_grad():
            outputs = self.spam_model(
                input_ids=tokenized_sent['input_ids'],
                attention_mask=tokenized_sent['attention_mask'],
                token_type_ids=tokenized_sent['token_type_ids']
                )

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits)
        return result