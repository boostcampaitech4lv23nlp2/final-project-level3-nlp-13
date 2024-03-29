import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from soynlp.normalizer import repeat_normalize

from inference import inference


def anonymize_nickname(sentence):
    return re.sub(r"[가-힣a-zA-Z0-9]+님", "<account>님", sentence)


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_last_word_is_only_brace(sent):
    # 마지막 단어가 괄호로만 이루어진 경우 제거
    try:
        if sent[-1] == "(":
            sent = sent[:-1].strip()
    except:
        pass
    return sent


def remove_not_korean(sent):
    if not re.search(r"[가-힣]", sent):
        return ""
    return sent


def remove_eng_upper_later(sent):
    # 영어 대문자로 이루어진 단어 이후 모두 제거
    eng_upper_idx = re.search(r"[A-Z]+", sent)
    if eng_upper_idx:
        sent = sent[: eng_upper_idx.start()]
    return sent


def remove_hash_tag(sent):
    # 해쉬태그 제거
    return re.sub(r"#\w+", "", sent)


def preprocess(sent):
    """URL 제거
    Args:
        sent (str): 전처리할 문장
    Returns:
        sent (str): 전처리된 문장
    """
    # 1. repeat_normalize
    sent = repeat_normalize(sent, num_repeats=3)

    # 2. ???님 => <account>님 으로 비식별화
    sent = anonymize_nickname(sent)

    # 3. url 제거
    sent = re.sub(r"http\S+", "", sent)
    sent = re.sub(r"https\S+", "", sent)

    # 4. 아이폰 이모지 제거
    sent = remove_emoji(sent)

    # 5. …가 붙어있는 단어 제거
    sent = re.sub(r"[가-힣a-zA-Z0-9-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'》]+…+", "", sent)

    # 6. 마지막 단어가 괄호로만 이루어진 경우 제거
    sent = remove_last_word_is_only_brace(sent)

    # &gt; &lt;
    sent = re.sub(r"&gt;", "", sent)
    sent = re.sub(r"&lt;", "", sent)

    # &amp; => &
    sent = re.sub(r"&amp;", "&", sent)

    # 한국어가 아닌 다른 언어의 트윗일 경우 제거
    sent = remove_not_korean(sent)

    # 해쉬태그 제거
    sent = remove_hash_tag(sent)

    # 영어 대문자로 이루어진 단어 제거 (DIOR, GLOBAL, JIMIN)
    sent = remove_eng_upper_later(sent)

    # 앞 뒤 공백 제거
    sent = sent.strip()

    return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="tweet_classification_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./corpus/twitter_classification/{args.config}.yaml")

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    ### 데이터 프레임 Q / A 형태 ###
    df = pd.DataFrame(columns=["Q", "A"])

    for file in os.listdir(config.data.twitter_pickle_path):
        if file.endswith(".pickle"):
            print(file)
            with open(os.path.join(config.data.twitter_pickle_path, file), "rb") as fr:
                data = pickle.load(fr)
                for key, value in data.items():
                    question = preprocess(value["question"])
                    if question == "":
                        continue

                    answers = value["answer"]
                    for answer in answers:
                        answer = preprocess(answer)
                        if answer == "":
                            continue
                        df = pd.concat([df, pd.DataFrame({"Q": [question], "A": [answer]})], ignore_index=True)

    # URL을 제거했을 때 공백인 경우(question, answer) 제거
    df = df[df["Q"] != ""]
    df = df[df["A"] != ""]

    # Q와 A가 완전 동일한 데이터 제거
    df["Q+A"] = df["Q"] + df["A"]
    df = df.drop_duplicates(["Q+A"])
    df = df.drop(["Q+A"], axis=1)

    # Q가 중복되는 것이 5개 이상인 경우 5개까지만 사용
    df = df.groupby("Q").head(5)
    df = df.reset_index(drop=True)

    # 데이터프레임 csv로 저장
    df.to_csv(config.data.pickle_to_csv_path, index=False, encoding="utf-8-sig")

    # 사용할만한 데이터만 추출
    inference(config)

    # inference한 데이터 로드
    inf_df = pd.read_csv(config.data.inference_save_path, encoding="utf-8-sig")
    label_1_df = inf_df[inf_df["pred"] == 1]
    label_1_df = label_1_df[["Q", "A"]]

    # 최종 데이터 저장
    if not os.path.exists("./data/processed_data/twitter/final"):
        os.makedirs("./data/processed_data/twitter/final")
    label_1_df.to_csv(config.data.final_save_path, index=False, encoding="utf-8-sig")
