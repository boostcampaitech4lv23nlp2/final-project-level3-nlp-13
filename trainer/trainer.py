import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import omegaconf
from datasets import Dataset, DatasetDict, load_metric
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class GPT_Chatbot:
    config: omegaconf.dictconfig.DictConfig
    training_args: TrainingArguments
    tokenizer: AutoTokenizer
    model: AutoModelForQuestionAnswering
    datasets: Optional[DatasetDict] = None

    def __post_init__(self):
        if self.datasets is not None:
            self.mode = "train"
            self.train_dataset = self.datasets.tokenized_datasets["train"]
            self.test_dataset = self.datasets.tokenized_datasets["test"]
        else:
            self.mode = "predict"

        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        # Trainer 초기화
        if self.mode == "train":
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.test_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.callbacks.early_stopping_patience)],
            )
        else:
            # inference
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )

    def train(self, checkpoint=None):
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(self.train_dataset)

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        output_train_file = os.path.join(self.training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"{key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        self.trainer.state.save_to_json(os.path.join(self.training_args.output_dir, "trainer_state.json"))
