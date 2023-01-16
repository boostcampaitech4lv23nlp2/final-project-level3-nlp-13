from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from app.model import Chatbot_utils, get_model
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from transformers import GPT2LMHeadModel

app = FastAPI()


class User_input(BaseModel):
    sentence: str


@app.post("/input", description="주문을 요청합니다")
async def make_chat(data: User_input, model: GPT2LMHeadModel = Depends(get_model)):
    generator = Chatbot_utils(model=model[0], tokenizer=model[1])
    text = data.dict()["sentence"]
    inference_result = generator.get_answer(text)
    # product = InferenceTextProduct(result=inference_result)
    # products.append(product)

    # new_order = Order(products=products)
    # texts.append(new_order)
    return inference_result
