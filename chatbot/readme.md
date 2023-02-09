## generator 모델 finetuning 실행 방법
```
python generator/train.py -c base_config
```
## generator 모델 pretraining 실행 방법
### BART
```
python generator/pretraining/run_bart_dlm_flax.py \
	--num_train_epochs=3.0 \
	--model_name_or_path="hyunwoongko/kobart" \
    --output_dir="saved_models/hyunwoongko/kobart" \
	--model_type="bart" \
    --tokenizer_name="hyunwoongko/kobart" \
	--dataset_name="oscar" \
    --max_seq_length="256" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
	--weight_decay="0.001" \
    --warmup_steps="30" \
    --logging_steps="10" \
    --save_steps="1" \
    --eval_steps="10"
```
### GPT
```
python generator/pretraining/run_clm_flax.py \
	--num_train_epochs=3.0
	--model_name_or_path="taeminlee/kogpt2" \
    --output_dir="saved_models/taeminlee/kogpt2" \
    --model_type="gpt2" \
    --tokenizer_name="taeminlee/kogpt2" \
    --dataset_name="oscar" \
	--do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-5" --warmup_steps="30" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --logging_steps="10" \
    --save_steps="1" \
    --eval_steps="10"
```
### T5
```
python generator/pretraining/run_t5_mlm_flax.py \
	--num_train_epochs=3.0 \
    --model_name_or_path="paust/pko-t5-base" \
	--output_dir="saved_models/paust/pko-t5-base" \
	--model_type="t5" \
	--tokenizer_name="paust/pko-t5-base" \
	--dataset_name="oscar" \
	--max_seq_length="256" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--weight_decay="0.001" \
	--warmup_steps="30" \
	--logging_steps="10" \
	--save_steps="1" \
	--eval_steps="10"
```

## retriever 모델 실행 방법
```
# elast_search 설치
bash install_elastic_search.sh
```

```
python retriever/elastic_retriever.py
```
