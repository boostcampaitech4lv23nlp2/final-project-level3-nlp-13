import argparse
import random

import numpy as np
import pandas as pd
import torch
from data_loader.data_loaders import ChatDataset, GPTDataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AdamW, DataCollatorForLanguageModeling, GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments
from utils.util import Chatbot_utils


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 device : ", device)

    print("🔥 get dataset...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        config.model.name, bos_token="</s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"
    )
    train_dataset = GPTDataset(tokenizer=tokenizer, file_path=config.path.train_path)

    print("🔥 get model...")
    model = GPT2LMHeadModel.from_pretrained(config.model.name)
    model.to("cuda")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print("🔥 start training...")
    args = TrainingArguments(
        output_dir="ex_kogpt2",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=config.train.max_epoch,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        save_steps=5_000,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset.tokenized_datasets["train"],
        # eval_dataset=tokenized_datasets["train"],
    )

    trainer.train()

    """ inference 부분으로 넣으면 될듯"""
    gen = Chatbot_utils(tokenizer, model)
    gen.get_answer("안녕?")
    gen.get_answer("만나서 반가워.")
    gen.get_answer("인공지능의 미래에 대해 어떻게 생각하세요?")
    gen.get_answer("여자친구 선물 추천해줘.")
    gen.get_answer("앞으로 인공지능이 어떻게 발전하게 될까요?")
    gen.get_answer("이제 그만 수업 끝내자.")


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
