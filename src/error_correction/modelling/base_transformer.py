#
# Copyright (c) 2019-2021 James Thorne.
#
# This file is part of factual error correction.
# See https://jamesthorne.co.uk for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import argparse
import os
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
    "error-correction": AutoModelForSeq2SeqLM,
}


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        mode="base",
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs
    ):

        super().__init__()
        self.hparams = hparams
        self.step_count = 0
        self.tfmr_ckpts = {}
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name
                if self.hparams.config_name
                else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name
                if self.hparams.tokenizer_name
                else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer

        self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer

        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def setup(self, step):
        train_batch_size = self.hparams.train_batch_size
        dataloader = self.get_dataloader("train", train_batch_size)
        self.train_loader = dataloader
        self.total_steps = (
            (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams.gpus)))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        save_path.mkdir(exist_ok=True)
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.tfmr_ckpts[self.step_count] = save_path

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="Linear warmup over warmup_steps.",
        )
        parser.add_argument(
            "--num_workers", default=4, type=int, help="kwarg passed to DataLoader"
        )
        parser.add_argument(
            "--num_train_epochs", dest="max_epochs", default=3, type=int
        )
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
