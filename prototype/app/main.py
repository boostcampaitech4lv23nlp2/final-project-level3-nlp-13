from app.model import Chatbot_utils, get_model
from fastapi import FastAPI
from fastapi.param_functions import Depends
from pydantic import BaseModel
from transformers import GPT2LMHeadModel

app = FastAPI()


class User_input(BaseModel):
    sentence: str
    max_len: int
    top_k: int
    top_p: float


@app.post("/input", description="주문을 요청합니다")
async def make_chat(data: User_input, model: GPT2LMHeadModel = Depends(get_model)):
    generator = Chatbot_utils(model=model[0], tokenizer=model[1])
    text = data.dict()["sentence"]
    max_len = data.dict()["max_len"]
    top_k = data.dict()["top_k"]
    top_p = data.dict()["top_p"]

    inference_result = generator.get_answer(text, max_len, top_k, top_p)

    return inference_result
