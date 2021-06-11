import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict
from pathlib import Path

from error_correction.modelling.utils import (
    encode_line,
    trim_batch,
    SortishSampler,
    recursive_clean,
)


class ErrorCorrectionSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        instance_generator,
        max_source_length,
        max_target_length,
        mutation_source,
        mutation_target,
        n_obs=None,
    ):
        super().__init__()
        self.instances = list(tqdm(filter(lambda i: i is not None, instance_generator)))
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mutation_src = mutation_source
        self.mutation_tgt = mutation_target
        self.has_preview = 0

    def __len__(self):
        return len(self.instances)

    def prepare_src(self, source, instance):
        if instance["evidence"] is None:
            return source

        if instance["mutation_type"] is not None and self.mutation_src:
            return (
                "claim: "
                + source
                + " "
                + "mutation: "
                + instance["mutation_type"]
                + " "
                + "evidence: "
                + instance["evidence"]
            )
        return "claim: " + source + " " + "evidence: " + instance["evidence"]

    def prepare_tgt(self, target, instance):
        if instance["mutation_type"] is not None and self.mutation_tgt:
            return (
                "mutation: " + instance["mutation_type"] + " " + "correction: " + target
            )

        return "correction: " + target

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        instance = self.instances[index]
        original_source_line = instance["source"]
        tgt_line = instance["target"]

        assert original_source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        masks = 0
        source_line = original_source_line
        while "[MASK]" in source_line:
            source_line = source_line.replace("[MASK]", f"<extra_id_{masks}>", 1)
            masks += 1

        source_input = self.prepare_src(source_line, instance)
        target_input = self.prepare_tgt(tgt_line, instance) + " </s>"

        source_inputs = encode_line(
            self.tokenizer, source_input, self.max_source_length
        )
        target_inputs = encode_line(
            self.tokenizer, target_input, self.max_target_length
        )
        original_ids = encode_line(self.tokenizer, source_line, self.max_source_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        original_ids = original_ids["input_ids"].squeeze()

        if self.has_preview < 5:
            self.has_preview += 1
            print(source_input)
            print(target_input)
            print(target_ids)
            print(recursive_clean({k: v for k, v in instance.items() if v is not None}))
            print("*" * 100)

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
            "original_ids": original_ids,
            "metadata": recursive_clean(
                {k: v for k, v in instance.items() if v is not None}
            ),
        }

        if self.has_preview < 5:
            self.has_preview += 1
            print(source_input)
            print(target_input)
            print(target_ids)
            print(recursive_clean({k: v for k, v in instance.items() if v is not None}))
            print("*" * 100)

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
            "original_ids": original_ids,
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
        originals = trim_batch(batch["original_ids"], pad_token_id)
        return source_ids, source_mask, y, originals

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        originals = torch.stack([x["original_ids"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        originals = trim_batch(originals, pad_token_id)
        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
            "original_ids": originals,
            "metadata": [x["metadata"] for x in batch if x is not None],
        }

        return batch

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)
