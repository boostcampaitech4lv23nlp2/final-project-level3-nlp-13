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
    gen_num = 2
    inputs = [
        "오늘 날씨 좋다",
        "얼어죽겠다",
        "너무 추워",
        "뭐해?",
        "밥 먹었어?",
        "잘자",
        "맛집 추천해주라",
        "배고프다",
        "졸립다",
        "나 고민이 있어",
        "개빡쳐",
        "우울하다",
        "행복해",
        "텅장됐다",
        "사랑해",
        "BTS 데뷔 언제 했어?",
        "진 전역일 언제야?",
        "너가 입덕한 이유는 뭐야?",
        "너는 최애가 누구야?",
        "옛날에 정국이가 추천했던 트로피컬하우스 노래 뭐였지",
        "석진이 오늘 뜬거에서 한 게임 뭐야?",
        "이번달방에서 했던게임 뭔지 아는사람?",
        "대상 슈상소감했어?",
        "최애 해메코 언제야?",
        "정구기 매력 세가지는?",
        "우리 호비만 귀 안뚫은거 맞지?",
        "지금 인디고 포카 교환 안구해지겠지?",
        "아 나 이거보고 우는중",
        "쏘왓 이부분은 진짜 눈이나 귀나 다 시원~함",
        "안이 이거 너무 귀엽다 ㅋㅋㅋㅋㅋㅋㅋㅋ",
        "혐생 살다와서 일상석진 지금 봄 ㅜㅜ",
        "태형이 주량 궁금하다",
        "얘들아 리플렉션 김남준 사랑해 떼창 쿨타임 찼다",
        "햇빛 보면서 맘마 먹는 햄찌니 봐 ㅋㅋㅋ",
        "김남준 갱장히 대학생처럼 하고 알쓸인잡 찍었네 ...",
        "눈뜨자마자 석찌일상 영상 보는나",
        "석지니는 진짜 선물만 주는구나ㅠㅠㅠ",
        "방금 골디에 홉이 잡혔었어!!!!",
        "정구기 처음으로 미워! 라는 단어 내뱉으니까 형아들 놀라 죽을라함 이런단어를 어디서 배웠냐고 소리꽥 지르는데 강아지유치원에서 친구가 쓰는거 들엇다고하는 꾹티즈",
        "찾았다..뱁새전정국은미쳤다\n얘표정좀봐 ㅁㅊ 무대한정변태맞다고",
        "이렇게 노메에 수수한 사복착장으로 저렇게 웃으면 유죄인간 하지만 정국이는 무조건 무죄",
        "아기톡히 오른쪽 왼쪽 껑충껑충 \n뛰넌거봐 졸귀야",
        "참나ㅋㅋ 석진이 게임한다고 입 앙다문 표정 좀 봐",
        "방탄 콘서트 당첨 될 확률 어느정도야?",
        "나 오늘 방탄 입덕한지 100일 됐어!",
        "BTS 노래 추천해줘",
        "달려라방탄 28/30일에 왜 안했어?",
        "보라해가 뭐야?",
        "BTS 콘서트 가고싶다",
        "김태형이랑 결혼하는 방법 알려줘",
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
