import argparse
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from tokenizers import SentencePieceBPETokenizer
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, PreTrainedTokenizerFast
from utils.util import Chatbot_utils


def main(config):
    print("🔥 get model...")
    if "gpt" in config.model.name_or_path:
        print("🔥 gpt")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            config.model.name_or_path,
            bos_token="</s>",
            eos_token="</s>",
            sep_token="<sep>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
        model = GPT2LMHeadModel.from_pretrained(config.model.name_or_path)
        model.resize_token_embeddings(len(tokenizer))
    elif (
        "bart" in config.model.name_or_path
        or "bart".upper() in config.model.name_or_path
        or "t5" in config.model.name_or_path
        or "t5".upper() in config.model.name_or_path
    ):
        print("🔥 Enc-Dec")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model.name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model.name_or_path)
        model.resize_token_embeddings(len(tokenizer))
    model.to("cuda")

    print("🔥 get input...")
    generator = Chatbot_utils(config, tokenizer, model)
    gen_num = 5
    generator.get_answer("안녕?", gen_num, config.tokenizer.max_length)
    generator.get_answer("만나서 반가워.", gen_num, config.tokenizer.max_length)
    generator.get_answer("인공지능의 미래에 대해 어떻게 생각하세요?", gen_num, config.tokenizer.max_length)
    generator.get_answer("여자친구 선물 추천해줘.", gen_num, config.tokenizer.max_length)
    generator.get_answer("앞으로 인공지능이 어떻게 발전하게 될까요?", gen_num, config.tokenizer.max_length)
    generator.get_answer("이제 그만 수업 끝내자.", gen_num, config.tokenizer.max_length)


if __name__ == "__main__":
    # config 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")

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
