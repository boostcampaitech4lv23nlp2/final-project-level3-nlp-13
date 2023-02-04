import torch
from transformers import AutoModelForSeq2SeqLM, GPT2LMHeadModel, PreTrainedTokenizerFast


class Generator:
    def __init__(self, config):
        self.config = config
        self.get_model()

    def get_model(self):
        model_path: str = self.config.model.name_or_path
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
            if "pretraining" in model_path:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, from_flax=True)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                model.resize_token_embeddings(len(tokenizer))
        model.to("cuda")
        self.tokenizer = tokenizer
        self.model = model

    def encoding(self, text):
        if "gpt" in self.config.model.name_or_path:
            text = self.tokenizer.bos_token + text + self.tokenizer.sep_token
        elif (
            "bart" in self.config.model.name_or_path
            or "bart".upper() in self.config.model.name_or_path
            or "t5" in self.config.model.name_or_path
            or "t5".upper() in self.config.model.name_or_path
        ):
            # text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
            pass
        return torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).to("cuda")

    def decoding(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def get_answer(self, input_sent, gen_num, max_len, top_k=50, top_p=0.95):
        input_sent = input_sent.strip()
        input_ids = self.encoding(input_sent)

        e_s = self.tokenizer.eos_token_id
        unk = self.tokenizer.unk_token_id

        sample_outputs = self.model.generate(
            input_ids,
            num_return_sequences=gen_num,
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

        print(f"input sentence : {input_sent}")
        for result in decoded_result:
            print(result)
        print()

        if len(decoded_result) == 1:
            return decoded_result[0]
        return decoded_result
