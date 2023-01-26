import argparse
import datetime

import numpy as np
import pytz
import torch
from datasets import load_dataset
from dpr_dataset import DPRDataset
from dpr_model import DPREncoder
from dpr_trainer import DPRTrainer
from omegaconf import OmegaConf
from transformers import BertTokenizer, TrainingArguments


def main(config):

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model.name_or_path)

    # load data from huggingface dataset
    data = load_dataset(config.data.path)
    train_data = data["train"]
    valid_data = data["test"]

    db_questions_data = train_data["Q"] + valid_data["Q"]
    db_answers_data = train_data["A"] + valid_data["A"]

    # make dataset
    # 현재는 test 용으로 train_data["Q"], train_data["A"]를 받음
    # 그러나 실제로는 들어오는 Query와 train_data["Q"]를 비교하여 가장 유사한 문장을 찾도록 학습해야함
    train_dataset = DPRDataset(train_data["Q"], train_data["A"], tokenizer)  # question과 answer의 input_ids, attention_mask, token_type_ids를 반환
    db_dataset = DPRDataset(db_questions_data, db_answers_data, tokenizer)

    # load p_encoder, q_encoder
    p_encoder = DPREncoder.from_pretrained(config.model.name_or_path)
    q_encoder = DPREncoder.from_pretrained(config.model.name_or_path)
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    # train
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    training_args = TrainingArguments(
        output_dir=f"./retriever/saved_models/dpr/encoder/{config.model.name_or_path}/{now_time}",
        evaluation_strategy="epoch",
        learning_rate=config.trainer.learning_rate,
        per_device_train_batch_size=config.trainer.batch_size,
        per_device_eval_batch_size=config.trainer.batch_size,
        num_train_epochs=config.trainer.num_train_epochs,
        weight_decay=config.trainer.weight_decay,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
    )

    trainer = DPRTrainer(training_args, config, tokenizer, p_encoder, q_encoder, train_dataset, db_dataset, train_data, valid_data)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="retriever_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./retriever/{args.config}.yaml")

    # fix random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    main(config)
