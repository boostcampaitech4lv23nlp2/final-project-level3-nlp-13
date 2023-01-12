import torch
import argparse
import random
import numpy as np
import pandas as pd
from data_loader.data_loaders import ChatDataset
from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, GPT2LMHeadModel, PreTrainedTokenizerFast
from omegaconf import OmegaConf

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 device : ", device)

    print("🔥 get dataset...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token="</s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"
)
    train_dataset = ChatDataset(tokenizer=tokenizer, file_path="data/ChatbotData.csv")
    data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print("🔥 get model...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token="</s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"
)
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    model.to("cuda")

    print("🔥 start training...")
    # get function handles of loss and metrics
    optimizer = AdamW(model.parameters(), lr=1e-4, correct_bias=True)
    epochs = 2
    avg_loss = (0.0, 0.0)

    for epoch in range(epochs):
    # for epoch in tqdm(range(epochs)):
        count = 0
        for data in data_loader:
            optimizer.zero_grad()
            # data = data.transpose(1,0)
            data = data[0].to("cuda")
            # model = model.to('cuda')

            outputs = model(data, labels=data)
            loss, logits = outputs[:2]
            loss = loss.to("cuda")
            loss.backward()
            avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
            optimizer.step()
            if count % 200 == 0:
                print(
                    "epoch no.{0}  train ({1}/{2})  loss = {3:.5f}  avg_loss = {4:.5f}".format(
                        epoch, count, len(data_loader), loss, avg_loss[0] / avg_loss[1]
                    )
                )
            count += 1
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    torch.save(model.state_dict(), "chitchat_model.bin")

if __name__ == '__main__':
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