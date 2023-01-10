import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token="</s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"
)
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
model.to("cuda")


def encoding(text):
    text = "<s>" + text + "</s><s>"
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0).to("cuda")


def decoding(ids):
    return tokenizer.decode(ids)


def get_answer(input_sent):
    input_ids = encoding(input_sent)

    e_s = tokenizer.eos_token_id
    unk = tokenizer.unk_token_id

    sample_outputs = model.generate(
        input_ids,
        num_return_sequences=5,
        do_sample=True,
        max_length=128,
        top_k=50,
        top_p=0.95,
        eos_token_id=e_s,
        early_stopping=True,
        bad_words_ids=[[unk]],  # 입력한 토큰(unk 토큰)이 생성되지 않도록 피하는 과정이 generate 함수 내에서 이루어짐
    )

    decoded_result = []
    for sample in sample_outputs:
        decoded_result.append(decoding(sample))
    for result in decoded_result:
        print(result)


get_answer("안녕?")
