import argparse
import os
import random

import numpy as np
import torch
from chatbot.generator.util import Generator
from omegaconf import OmegaConf


def main(config):
    print("🔥 get model...")
    generator = Generator(config)

    print("🔥 get input...")
    gen_num = 5
    inputs = [
        "안녕?",
        "만나서 반가워.",
        "인공지능의 미래에 대해 어떻게 생각하세요?",
        "여자친구 선물 추천해줘.",
        "앞으로 인공지능이 어떻게 발전하게 될까요?",
        "이제 그만 수업 끝내자.",
        "아 전정국 땜에 괴롭다 귀에대고 들어봐",
        "이 시기를 지나온 선배아미님들 다시 한번 존경해야되는것같음 👍👍👍.",
        "영배 선배님… 부디 🙇🏻‍♀️ 뵤아리와 통화를…",
        "럽셀콘 많이가서 셋리를 다 외우고있을때가 있었는데… 3-4년 전이라는게 안믿김…",
        "👍👍👍역시ㅎㅎ 오십페이지 무슨 일이랍니까 기절ㅠ",
    ]
    for sent in inputs:
        generator.get_answer(sent, gen_num, config.tokenizer.max_length)


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
