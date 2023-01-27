import argparse
import datetime

import emoji
import numpy as np
import pandas as pd
import pytz
import torch
from dataset import TwitterClassificationDataset
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from soynlp.normalizer import repeat_normalize
from transformers import ElectraForSequenceClassification, ElectraTokenizer, Trainer, TrainingArguments


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    model = ElectraForSequenceClassification.from_pretrained(config.model.name_or_path, num_labels=3, problem_type="multi_label_classification")
    tokenizer = ElectraTokenizer.from_pretrained(config.model.name_or_path)

    # load csv
    df = pd.read_csv(config.data.train_data_path, encoding="utf-8")

    # split train and eval stratified
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label"], shuffle=True)
    train_df.reset_index(drop=True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)

    # load dataset
    train_dataset = TwitterClassificationDataset(train_df, tokenizer)
    eval_dataset = TwitterClassificationDataset(eval_df, tokenizer)

    # load trainer
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    training_args = TrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_ratio=0.5,
        weight_decay=0.01,
        learning_rate=5e-5,
        save_steps=500,
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=2,
        evaluation_strategy="steps",
        output_dir=f"./corpus/twitter_classification/saved_models/{config.model.name_or_path}/{now_time}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # train
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="tweet_classification_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./corpus/twitter_classification/{args.config}.yaml")

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    main(config)
