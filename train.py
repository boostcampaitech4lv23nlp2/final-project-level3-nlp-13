import argparse
import datetime

import numpy as np
import pytz
import torch
import wandb
from data_loader.data_loaders import ChatDataset, GPT_Dataset
from omegaconf import OmegaConf
from trainer.trainer import GPT_Chatbot
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, TrainingArguments


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 device : ", device)

    print("🔥 get dataset...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        config.model.name_or_path, bos_token="</s>", eos_token="</s>", sep_token="<sep>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"
    )
    train_dataset = GPT_Dataset(tokenizer=tokenizer, config=config)

    print("🔥 get model...")
    model = GPT2LMHeadModel.from_pretrained(config.model.name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to("cuda")

    print("🔥 start training...")
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    file_name = f"saved_models/{config.model.name_or_path}/{config.train.num_train_epochs}epoch_{now_time}"
    training_args = TrainingArguments(**config.train, output_dir=file_name)
    if config.wandb.use:
        run_id = f"chatbot_{config.wandb.name}_{now_time}"
        wandb.init(
            entity=config.wandb.team,
            project=config.wandb.project,
            group=config.model.name_or_path,
            id=run_id,
            tags=config.wandb.tags,
        )
        training_args.report_to = ["wandb"]

    trainer = GPT_Chatbot(
        config=config,
        training_args=training_args,
        tokenizer=tokenizer,
        model=model,
        datasets=train_dataset,
    )

    trainer.train()

    # share the pretrained model to huggingface hub
    if config.hf_hub.push_to_hub is True:
        save_name = config.hf_hub.save_name
        if not save_name.startswith("nlpotato/"):
            save_name = "nlpotato/" + save_name
        model.push_to_hub(config.hf_hub.save_name)
        tokenizer.push_to_hub(config.hf_hub.save_name)

    wandb.finish()


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
