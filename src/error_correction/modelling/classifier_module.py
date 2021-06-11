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
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from error_correction.modelling.base_transformer import BaseTransformer
from error_correction.modelling.dataset.classification_dataset import (
    FEVERClsDatasetNS,
    FEVERClsDataset,
)
from error_correction.modelling.lightning_base import add_generic_args
from error_correction.modelling.reader.fever_reader import (
    FEVERReader,
    FEVERReaderSubSample,
    FEVERReaderNonNeutral,
)
from error_correction.modelling.utils import (
    use_task_specific_params,
    pickle_save,
    assert_all_frozen,
    freeze_params,
    FastAccuracyEpoch,
    is_truthy,
    save_json,
    flatten_list,
    simple_accuracy,
)


class FEVERClassifierModule(BaseTransformer):
    mode = "sequence-classification"
    val_metric = "accuracy"
    loss_names = ["loss"]
    metric_names = ["accuracy"]

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=3, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "sequence-classification")

        self.metrics_save_path = Path(self.output_dir) / (
            "metrics.json"
            if not self.hparams.do_predict
            else "metrics_test_{}.json".format(os.path.basename(self.hparams.test_file))
        )
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            max_source_length=self.hparams.max_source_length,
            negative_sample=is_truthy(self.hparams.negative_sampling),
        )

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {
            k: v if v >= 0 else None for k, v in n_observations_per_split.items()
        }
        self.data_paths = {
            "train": self.hparams.train_file,
            "val": self.hparams.val_file,
            "test": self.hparams.test_file,
        }

        if self.hparams.freeze_embeds:
            self.freeze_embeds()

        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None
        self.dataset_class = (
            FEVERClsDatasetNS
            if self.dataset_kwargs["negative_sample"]
            else FEVERClsDataset
        )
        self.wiki_reader = self.get_reader(self.hparams.reader, self.hparams.do_predict)
        self.acc = FastAccuracyEpoch()

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        r"""
        Additional items to be displayed in the progress bar.

        Return:
            Dictionary with the items to be displayed in the progress bar.
        """
        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = (
            running_train_loss.cpu().item()
            if running_train_loss is not None
            else float("NaN")
        )
        tqdm_dict = {"loss": "{:.3f}".format(avg_training_loss)}

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict["split_idx"] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict["v_num"] = self.trainer.logger.version

        tqdm_dict["ac"] = (self.acc.hits / self.acc.total) if self.acc.total > 0 else 0

        return tqdm_dict

    def get_reader(self, name, test):
        if name == "fever":
            return FEVERReader(test)
        elif name == "fever_sub":
            return FEVERReaderSubSample(test)
        elif name == "fever_noneg":
            return FEVERReaderNonNeutral(test)
        else:
            raise RuntimeError(f"Unknown reader {name}")

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def _step(self, batch: dict) -> Tuple:
        input_ids, attn_mask, label_ids = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        outputs = self(input_ids=input_ids, attention_mask=attn_mask, labels=label_ids)
        # label_logits = outputs[0]
        #
        # # Same behavior as modeling_bart.py, besides ignoring pad_token_id
        # ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        #
        # loss = ce_loss_fct(label_logits, label_ids.view(-1))
        return outputs[0], outputs[1]

    # def training_step(self, batch, batch_idx):
    #     inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    #
    #     if self.config.model_type != "distilbert":
    #         inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None
    #
    #     outputs = self(**inputs)
    #     loss = outputs[0]
    #
    #     lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
    #     tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
    #     return {"loss": loss, "log": tensorboard_logs}

    def compute_metrics(self, label_logits, label_ids) -> Dict:
        return {
            "accuracy": simple_accuracy(
                np.argmax(label_logits.detach().cpu(), axis=1), label_ids.detach().cpu()
            )
        }

    def compute_metrics_fast(self, label_logits, label_ids) -> Dict:
        return {
            "accuracy": self.acc.fast_accuracy(
                np.argmax(label_logits.detach().cpu(), axis=1), label_ids.detach().cpu()
            )
        }

    def training_step(self, batch, batch_idx) -> Dict:
        loss, logits = self._step(batch)
        ad = self.compute_metrics_fast(logits, batch["labels"])
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        # print()
        # print(loss)
        # print(lr_scheduler.get_last_lr()[-1])
        # print(np.argmax(logits.detach().cpu(), axis=1))
        # print(batch["labels"].view(-1).cpu())
        # print()
        return {
            "loss": loss,
            "log": {
                "loss": loss,
                "accuracy": ad["accuracy"],
                "lr": lr_scheduler.get_last_lr()[-1],
            },
            "accuracy": ad["accuracy"],
        }

    def validation_step(self, batch, batch_idx) -> Dict:

        loss, logits = self._step(batch)

        ad = self.compute_metrics(logits, batch["labels"])
        return {
            "loss": loss,
            "log": {"loss": loss, "accuracy": ad["accuracy"]},
            "accuracy": ad["accuracy"],
        }

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {
            k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names
        }
        loss = losses["loss"]

        self.acc.reset()

        rouges = {
            k: np.array(flatten_list([x[k] for x in outputs])).mean().item()
            for k in self.metric_names
        }
        rouge_tensor: torch.FloatTensor = torch.tensor(rouges[self.val_metric]).type_as(
            loss
        )
        rouges.update({k: v.item() for k, v in losses.items()})
        losses.update(rouges)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count

        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path

        return {
            "log": metrics,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": rouge_tensor,
        }

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def save_predictions(self, predictions, actual, metadata, type_path) -> None:
        with open(
            Path(self.output_dir)
            / (
                "predictions_set_{}_epoch_{}_steps_{}_.jsonl".format(
                    type_path, self.trainer.current_epoch, self.step_count
                )
                if not self.hparams.do_predict
                else "final_predictions_set_{}_file_{}".format(
                    type_path, os.path.basename(self.hparams.test_file)
                )
            ),
            "w+",
        ) as f:
            for p, a, m in zip(predictions, actual, metadata):
                f.write(
                    json.dumps({"prediction": p, "actual": a, "metadata": m}) + "\n"
                )

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> FEVERClsDataset:
        n_obs = self.n_obs[type_path]
        instance_generator = self.wiki_reader.read(self.data_paths[type_path])

        dataset = self.dataset_class(
            self.tokenizer,
            instance_generator=instance_generator,
            n_obs=n_obs,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(
        self, type_path: str, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        dataset = self.get_dataset(type_path)
        sampler = None
        if self.hparams.sortish_sampler and type_path == "train":
            assert self.hparams.gpus <= 1  # TODO: assert earlier
            sampler = dataset.make_sortish_sampler(batch_size)
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader(
            "train", batch_size=self.hparams.train_batch_size, shuffle=True
        )
        t_total = (
            (
                len(dataloader.dataset)
                // (self.hparams.train_batch_size * max(1, self.hparams.gpus))
            )
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        if max(scheduler.get_last_lr()) > 0:
            warnings.warn("All learning rates are 0")
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)

        parser.add_argument("--reader", default="fever", type=str)
        parser.add_argument("--train_file", required=True, type=str)
        parser.add_argument("--val_file", required=True, type=str)
        parser.add_argument("--test_file", required=False, type=str)
        parser.add_argument(
            "--max_source_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument(
            "--logger_name",
            type=str,
            choices=["default", "wandb", "wandb_shared"],
            default="default",
        )
        parser.add_argument(
            "--n_train",
            type=int,
            default=-1,
            required=False,
            help="# examples. -1 means use all.",
        )
        parser.add_argument(
            "--n_val",
            type=int,
            default=-1,
            required=False,
            help="# examples. -1 means use all.",
        )
        parser.add_argument(
            "--n_test",
            type=int,
            default=-1,
            required=False,
            help="# examples. -1 means use all.",
        )
        parser.add_argument(
            "--negative_sampling", type=str, required=False, default="false"
        )
        return parser
