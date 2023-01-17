from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import datasets
import sys
import argparse
import pandas as pd
import numpy as np
import datetime
import pytz
from datasets import Dataset, DatasetDict,load_dataset
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from data_loader.data_loaders import SingleSentDataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

def main(config):
    print("🤬 get model...")
    tokenizer = AutoTokenizer.from_pretrained(config.spam.model.name)
    model = BertForSequenceClassification.from_pretrained(config.spam.path.model)

    def sentences_predict(sent):        
        model.eval()
        tokenized_sent = tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                add_special_tokens=True,
                max_length=128
        )
        #tokenized_sent.to(device)
        
        with torch.no_grad():# 그라디엔트 계산 비활성화
            outputs = model(
                input_ids=tokenized_sent['input_ids'],
                attention_mask=tokenized_sent['attention_mask'],
                token_type_ids=tokenized_sent['token_type_ids']
                )

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits)
        return result

    print(sentences_predict("짱개 주작 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ")) #1
    print(sentences_predict("어제 하루종일 속상하고 나니까")) #0
    print(sentences_predict("게이야 용기사면 드래곤 사역하고있노?")) #1
    print(sentences_predict("밝은 회색이 유행이라던데")) #0
    print(sentences_predict("지방것들이")) #1

if __name__ == "__main__":
    # config 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./config/{args.config}.yaml")

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    main(config)