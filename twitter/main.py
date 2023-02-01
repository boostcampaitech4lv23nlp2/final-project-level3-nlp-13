import tweepy
import time
import random
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from tokenizers import SentencePieceBPETokenizer
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, PreTrainedTokenizerFast, BertForSequenceClassification
from utils.util import Chatbot_utils


load_dotenv()
TWITTER_CONSUMER_KEY = os.environ.get("CONSUMER_KEY")
TWITTER_CONSUMER_SECRET_KEY = os.environ.get("CONSUMER_SECRET")
TWITTER_ACCESS_TOKEN = os.environ.get("ACCESS_KEY")
TWITTER_ACCESS_SECRET_TOKEN = os.environ.get("ACCESS_SECRET")

auth  = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET_KEY)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET_TOKEN)
 
api = tweepy.API(auth, wait_on_rate_limit=True)
 
FILE_NAME = 'last_seen_id.txt'
 
def retrieve_last_seen_id(file_name):
    f_read = open(file_name, 'r')
    last_seen_id = int(f_read.read().strip())
    f_read.close()
    return last_seen_id
 
def store_last_seen_id(last_seen_id, file_name):
    f_write = open(file_name, 'w')
    f_write.write(str(last_seen_id))
    f_write.close()
    return

def sentences_predict(sent):        
    spam_model.eval()
    tokenized_sent = spam_tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
    )

    with torch.no_grad():
        outputs = spam_model(
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits)
    return result

def reply_to_tweets():
    global i, o
    print('🔥 트윗 확인 중...', flush=True)
    last_seen_id = retrieve_last_seen_id(FILE_NAME)
    mentions = api.mentions_timeline(last_seen_id,tweet_mode='extended')
    #print("💭mentions:", mentions)

    for mention in reversed(mentions):
        last_seen_id = mention.id
        store_last_seen_id(last_seen_id, FILE_NAME)

        username = '@ja_smilee' #🔥 username 바꾸기
        if username in mention.full_text.lower(): 
            # 1. 멘션당함
            input_text = mention.full_text.replace(str(mention.user.screen_name), '').replace("@",'')
            print("👋 tweet_id:", str(mention.user.screen_name) + '/ 👋 tweet_text:' + input_text, flush=True) 
            
            # 🤬 2. spam filtering step 🤬
            if sentences_predict(input_text) == 1:
                print("🤬It is a spam🤬")
                output_text = "바르고 고운 말만 해주세요 감사합니다"
            else:
                print("⚫ Not spam. generate answer for...", input_text) 
                generator = Chatbot_utils(tokenizer, model)
                output_text = generator.get_answer(input_text)
                output_text = str(output_text).replace(input_text[1:], "")
                while True:
                    if sentences_predict(output_text) == 0:
                        print("❌햄 output: ", output_text)
                        break
                    print("⭕ 스팸: ", sentences_predict(output_text))
                    output_text = generator.get_answer(input_text)
                    output_text = str(output_text).replace(input_text[1:], "")

            # 3. 답글 업로드
            new_status = api.update_status("@"+ mention.user.screen_name + " " + output_text, mention.id) 
            print("💬 retweeted_id:", "@"+ mention.user.screen_name , "/ 💬 retweet_text: ", output_text)
            print()

if __name__ == "__main__":
    # config 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")
    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./config/{args.config}.yaml")

    print("🔥 get Chatbot model...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        config.model.name, bos_token="</s>", eos_token="</s>", sep_token="<sep>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"
    )
    model = GPT2LMHeadModel.from_pretrained(config.model.name)
    model.resize_token_embeddings(len(tokenizer))
    model.to("cuda")
    
    print("🔥 get Spam filtering model...")
    spam_tokenizer = AutoTokenizer.from_pretrained(config.model.spam)
    spam_model = BertForSequenceClassification.from_pretrained(config.model.spam)

    while True:
        reply_to_tweets()
        time.sleep(5)
