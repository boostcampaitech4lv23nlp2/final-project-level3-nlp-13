import argparse
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import random
import numpy as np
import pandas as pd
from data_loader.data_loaders import ChatDataset
from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
import os


def get_answer(input_sent):
    # encoding
    text = "<s>" + input_sent + "</s><s>"
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to("cuda")

    e_s = tokenizer.eos_token_id
    unk = tokenizer.unk_token_id

    sample_outputs = model.generate(
        input_ids,
        num_return_sequences=5,
        do_sample=True,
        max_length=128,
        top_k=50,
        top_p=0.95,
        eos_token_id=e_s,
        early_stopping=True,
        bad_words_ids=[[unk]],  # 입력한 토큰(unk 토큰)이 생성되지 않도록 피하는 과정이 generate 함수 내에서 이루어짐
    )

    decoded_result = []
    for sample in sample_outputs:
        decoded_result.append(tokenizer.decode(sample)) # decoding
    for result in decoded_result:
        print(result)

def main(config):
    print("🔥 get model...")
    global tokenizer, model
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
    config.model.name, bos_token="</s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"
)
    model = GPT2LMHeadModel.from_pretrained(config.model.name)
    model.to("cuda")
    print("🔥 get input...")
    get_answer("안녕?")


if __name__ == "__main__":
    # config 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./config/{args.config}.yaml")

    # seed 설정
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(config)
