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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🤬 device : ", device)

    print("🤬 get dataset...")
    dataset = load_dataset(config.spam.path.train_path)

    # 필요한 데이터인 document와 label 정보만 pandas라이브러리 DataFrame 형식으로 변환
    train_data = pd.DataFrame({"document":dataset['train']['document'], "label":dataset['train'][' label'],})
    tokenizer = AutoTokenizer.from_pretrained(config.spam.model.name)
    tokenized_train_sentences = tokenizer(
        list(train_data['document']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        )
    train_dataset = SingleSentDataset(tokenized_train_sentences, train_data['label'].values)

    print("🤬 get model...")
    model = BertForSequenceClassification.from_pretrained(config.spam.model.name)
    model.to(device)

    print("🤬 start training...")
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    file_name = f"spam_saved_models/{config.spam.model.name}_{now_time}_{config.spam.train.max_epoch}epoch"

    # ⭐ 문장 분류를 위해선 BERT 위에 classification을 위한 head를 부착해야 합니다.
    # ⭐ 해당 부분을 transformers에서는 라이브러리 하나만 호출하면 됩니다! :-)
    training_args = TrainingArguments(
        output_dir=file_name,          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=500,
        save_steps=500,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
    )
    trainer.train()
    trainer.save_model(file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="base_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./config/{args.config}.yaml")

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    main(config)