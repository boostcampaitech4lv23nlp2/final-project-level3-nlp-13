import torch


class Chatbot_utils:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def encoding(self, text):
        text = "</s>" + text + "<sep>"
        return torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).to("cuda")

    def decoding(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def get_answer(self, input_sent):
        input_ids = self.encoding(input_sent)

        e_s = self.tokenizer.eos_token_id
        unk = self.tokenizer.unk_token_id

        sample_outputs = self.model.generate(
            input_ids,
            num_return_sequences=1,
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
            decoded_result.append(self.decoding(sample))

        return decoded_result[0] #[0].replace(input_sent, "") 

        '''
        for result in decoded_result:
            print(result.replace("<sep>", ""))  # special token 처리를 해주지 않았기 때문에 후처리 필요
            print()
        '''
