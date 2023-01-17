import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


def get_model(
    model_path: str = "nlpotato/kogpt2_chatbot_social_media-e10",
) -> GPT2LMHeadModel:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        # "skt/kogpt2-base-v2",
        bos_token="</s>",
        eos_token="</s>",
        sep_token="<sep>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to("cuda")
    return model, tokenizer


class Chatbot_utils:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def encoding(self, text):
        text = "</s>" + text + "<sep>"
        return torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).to("cuda")

    def decoding(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def get_answer(self, input_sent, max_len, top_k, top_p):
        input_ids = self.encoding(input_sent)

        e_s = self.tokenizer.eos_token_id
        unk = self.tokenizer.unk_token_id

        sample_outputs = self.model.generate(
            input_ids,
            num_return_sequences=1,
            do_sample=True,
            max_length=max_len,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=e_s,
            early_stopping=True,
            bad_words_ids=[[unk]],  # 입력한 토큰(unk 토큰)이 생성되지 않도록 피하는 과정이 generate 함수 내에서 이루어짐
        )

        decoded_result = []
        for sample in sample_outputs:
            decoding = self.decoding(sample)
            decoded_result.append(decoding.replace(input_sent, ""))

        for result in decoded_result:
            print(result)

        if len(decoded_result) == 1:
            return decoded_result[0]
        return decoded_result
