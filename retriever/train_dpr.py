import argparse
import datetime

import numpy as np
import pandas as pd
import pytz
import torch
from datasets import load_dataset
from dpr_dataset import DPRDataset
from dpr_model import DPREncoder, DPRModel
from omegaconf import OmegaConf
from transformers import BertTokenizer, DPRQuestionEncoder, PreTrainedModel, TrainingArguments


def main(config):
    tokenizer = BertTokenizer.from_pretrained(config.model.name_or_path)

    # load data from huggingface dataset
    data = load_dataset(config.data.path)
    train_data = data["train"]
    valid_data = data["test"]

    # make dataset
    train_dataset = DPRDataset(train_data["Q"], train_data["A"], tokenizer)  # question과 answer의 input_ids, attention_mask, token_type_ids를 반환
    valid_dataset = DPRDataset(valid_data["Q"], valid_data["A"], tokenizer)

    sent = train_dataset[0]
    print(sent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="retriever_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./retriever/{args.config}.yaml")

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    main(config)
