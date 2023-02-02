import torch
from transformers import AutoModelForSeq2SeqLM, GPT2LMHeadModel, PreTrainedTokenizerFast


def get_model(
    model_path: str = "nlpotato/pko-t5-base-pretraining_finetuning_temp1",
):
    if "gpt" in model_path:
        print("🔥 gpt")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_path,
            bos_token="</s>",
            eos_token="</s>",
            sep_token="<sep>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
    elif "bart" in model_path or "bart".upper() in model_path or "t5" in model_path or "t5".upper() in model_path:
        print("🔥 enc-dec")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
    model.to("cuda")
    return model, tokenizer
