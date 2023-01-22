import argparse
import sys

from fastapi import FastAPI
from fastapi.param_functions import Depends
from omegaconf import OmegaConf
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM

from .model import get_model

sys.path.append("..")  # Adds higher directory to python modules path.
from utils.util import Chatbot_utils

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="base_config")
args, _ = parser.parse_known_args()
config = OmegaConf.load(f"../config/{args.config}.yaml")

app = FastAPI()


class User_input(BaseModel):
    sentence: str
    max_len: int
    top_k: int
    top_p: float


@app.post("/input", description="주문을 요청합니다")
async def make_chat(data: User_input, model: AutoModelForSeq2SeqLM = Depends(get_model)):
    model = get_model(config.model.name_or_path)
    generator = Chatbot_utils(config, model=model[0], tokenizer=model[1])
    text = data.dict()["sentence"]
    max_len = data.dict()["max_len"]
    top_k = data.dict()["top_k"]
    top_p = data.dict()["top_p"]

    inference_result = generator.get_answer(text, 1, max_len, top_k, top_p)

    return inference_result
