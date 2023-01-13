import argparse
import datetime

import numpy as np
import pytz
import torch
from data_loader.data_loaders import ChatDataset, GPTDataset
from omegaconf import OmegaConf
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 device : ", device)

    print("🔥 get dataset...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        config.model.name, bos_token="</s>", eos_token="</s>", sep_token="<sep>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"
    )
    train_dataset = GPTDataset(tokenizer=tokenizer, file_path=config.path.train_path)

    print("🔥 get model...")
    model = GPT2LMHeadModel.from_pretrained(config.model.name)
    model.resize_token_embeddings(len(tokenizer))
    model.to("cuda")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print("🔥 start training...")
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    file_name = f"saved_models/{config.model.name}_{now_time}_{config.train.max_epoch}epoch"

    args = TrainingArguments(
        output_dir=file_name,
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
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset.tokenized_datasets["train"],
        # eval_dataset=tokenized_datasets["train"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
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
