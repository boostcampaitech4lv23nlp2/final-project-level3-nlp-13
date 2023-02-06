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
from data_loader.data_loaders import ClassificationDataset
from transformers import (
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    ElectraForSequenceClassification,
    ElectraTokenizer,
    DataCollatorForTokenClassification,
    )
    
def main(config):
    if config.spam.model.name =="klue/bert-base":
        print("🤬 get BERT model...")
        tokenizer = AutoTokenizer.from_pretrained(config.spam.path.inference_model)
        model = BertForSequenceClassification.from_pretrained(config.spam.path.inference_model)
    else:
        print("🤬 get Electra model...")
        tokenizer = ElectraTokenizer.from_pretrained(config.spam.path.inference_model)
        model = ElectraForSequenceClassification.from_pretrained(config.spam.path.inference_model)
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
        return sent, result

    print(sentences_predict("시발")) #1
    print(sentences_predict("시발롬아")) #1
    print(sentences_predict("ㅅㅂ")) #1
    print(sentences_predict("지랄")) #1
    print(sentences_predict("ㅈㄹ")) #1
    print(sentences_predict("존나 무섭네")) #1
    print(sentences_predict("개꼴 씨발 ㅋㅋ")) #1
    print(sentences_predict("시발 일게이들 이제 러브인 아시아 강제로 찍게되노 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ")) #1
    print(sentences_predict("이미 한남 와꾸는........ 읍음....!!!!")) #1
    print(sentences_predict("게이도 좆같고 레즈도 좆같고 동성애 씨발년놈들은 옛날처럼 아우슈비츠 가스실에 싸그리 모아놓고 독가스 살포해서애미애비 보는앞에서 고통스럽고 잔인하게 죽여야된다")) #1
    print(sentences_predict("짱개 주작 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ")) #1
    print(sentences_predict("게이야 용기사면 드래곤 사역하고있노?")) #1
    print(sentences_predict("지방것들이")) #1
    print(sentences_predict("ㄹㅇ 개좃같은 페미민국 진짜요즘 좌좀 달창 분탕년들 남초에 오지게 풀었더만애국하면서 노예처럼 살아가라고 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ")) #1
    print(sentences_predict("속상하고 빡쳐")) #1

    print(sentences_predict("속상하고 짜증나 죽겠음 진짜")) #0
    print(sentences_predict("힝 불쌍해")) #0
    print(sentences_predict("어제 하루종일 속상하고 나니까")) #0
    print(sentences_predict("밝은 회색이 유행이라던데")) #0
    print(sentences_predict("뭐 어쩌라고")) #0
    print(sentences_predict("페미니스트의 모습이다")) #0
    print(sentences_predict("졸라 귀엽다....")) #0
    print(sentences_predict("세상이 밉다")) #0
    print(sentences_predict("항상 댓글들이 왜 이럴까 의아했는데 지금은 상황이 이렇게 바뀌다니. 중국 참 무섭고 화나네.")) #0

    
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
