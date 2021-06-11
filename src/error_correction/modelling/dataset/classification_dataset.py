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
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict
from pathlib import Path

from error_correction.modelling.utils import (
    encode_line,
    recursive_clean,
    trim_batch,
    SortishSampler,
)


class FEVERClsDataset(Dataset):
    LABELS = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

    def __init__(
        self,
        tokenizer,
        instance_generator,
        max_source_length,
        n_obs=None,
    ):
        super().__init__()
        self.instances = list(tqdm(filter(lambda i: i is not None, instance_generator)))
        self.max_source_length = max_source_length
        self.tokenizer = tokenizer
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.has_preview = 0
        self.labels = dict()

    def __len__(self):
        return len(self.instances)

    def prepare_src(self, source, instance):
        return source + " " + self.tokenizer.sep_token + " " + instance["evidence"]

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        instance = self.instances[index]
        original_source_line = instance["source"]
        assert original_source_line, f"empty claim index {index}"

        return self.process_item(instance)

    def process_item(self, instance):
        source_line = instance["source"]
        source_input = self.prepare_src(source_line, instance)
        source_inputs = encode_line(
            self.tokenizer, source_input, self.max_source_length
        )

        source_ids = source_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        type_ids = source_inputs["token_type_ids"].squeeze()
        label_id = self.add_or_get(instance["label"])

        if self.has_preview < 5:
            self.has_preview += 1
            print(source_input)
            print(source_ids)
            print(self.tokenizer.convert_ids_to_tokens(source_ids))

            print(label_id)
            print(recursive_clean({k: v for k, v in instance.items() if v is not None}))
            print("*" * 100)

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "type_ids": type_ids,
            "label": label_id,
            "metadata": recursive_clean(
                {k: v for k, v in instance.items() if v is not None}
            ),
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id) -> tuple:
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(
            batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"]
        )
        return source_ids, source_mask, y

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        labels = torch.LongTensor([x["label"] for x in batch])
        pad_token_id = self.pad_token_id
        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )

        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": labels,
            "metadata": [x["metadata"] for x in batch if x is not None],
        }
        return batch

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)

    def add_or_get(self, label):
        return self.LABELS[label]


class FEVERClsDatasetNS(Dataset):
    LABELS = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

    def __init__(
        self,
        tokenizer,
        instance_generator,
        max_source_length,
        negative_sample=False,
        n_obs=None,
    ):
        super().__init__()
        self.instances = list(tqdm(filter(lambda i: i is not None, instance_generator)))

        if negative_sample:  # and not instance_generator.test:
            negative_instances = []
            self.random = random.Random(1)
            for instance in self.instances:
                if self.random.uniform(0, 1) > 0.5:
                    continue
                negative_inst = copy(instance)
                rand_inst = self.instances[
                    self.random.randint(0, len(self.instances) - 1)
                ]
                negative_inst["evidence"] = rand_inst["evidence"]
                negative_inst["label"] = "NOT ENOUGH INFO"
                negative_instances.append(negative_inst)

            self.instances.extend(negative_instances)
            self.random.shuffle(self.instances)

        self.max_source_length = max_source_length
        self.tokenizer = tokenizer
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.has_preview = 0
        self.labels = dict()

    def __len__(self):
        return len(self.instances)

    def prepare_src(self, source, instance):
        return source + " " + self.tokenizer.sep_token + " " + instance["evidence"]

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        instance = self.instances[index]
        original_source_line = instance["source"]
        assert original_source_line, f"empty claim index {index}"

        source_line = original_source_line
        source_input = self.prepare_src(source_line, instance)
        source_inputs = encode_line(
            self.tokenizer, source_input, self.max_source_length
        )

        source_ids = source_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        type_ids = source_inputs["token_type_ids"].squeeze()
        label_id = self.add_or_get(instance["label"])

        if self.has_preview < 5:
            self.has_preview += 1
            print(source_input)
            print(source_ids)
            print(self.tokenizer.convert_ids_to_tokens(source_ids))

            print(label_id)
            print(recursive_clean({k: v for k, v in instance.items() if v is not None}))
            print("*" * 100)

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "type_ids": type_ids,
            "label": label_id,
            "metadata": recursive_clean(
                {k: v for k, v in instance.items() if v is not None}
            ),
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id) -> tuple:
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(
            batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"]
        )
        return source_ids, source_mask, y

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        type_ids = torch.stack([x["type_ids"] for x in batch])
        labels = torch.LongTensor([x["label"] for x in batch])
        pad_token_id = self.pad_token_id
        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )
        # type_ids, _ = trim_batch(type_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            # "token_type_ids": type_ids,
            "labels": labels,
            "metadata": [x["metadata"] for x in batch if x is not None],
        }
        return batch

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)

    def add_or_get(self, label):
        return self.LABELS[label]
        # if label in self.labels:
        #     return self.labels[label]
        # else:
        #     self.labels[label] = len(self.labels)
        #     return self.labels[label]
