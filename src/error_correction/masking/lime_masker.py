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
import itertools
import argparse
import json
from copy import copy
from typing import List, Iterator

import torch
import numpy as np
from collections import Iterable
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm


from transformers import AutoModelForSequenceClassification, AutoTokenizer

from error_correction.masking.json_encoder import NpEncoder
from error_correction.modelling.dataset.classification_dataset import FEVERClsDataset
from error_correction.modelling.reader.fever_reader import FEVERReader


def lazy_groups_of(iterable: Iterable, group_size: int) -> Iterator[List]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(itertools.islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


def move(dict_of_tensors, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in dict_of_tensors.items()
    }


def model_predict(model, batch):
    input_ids, attn_mask, label_ids = (
        batch["input_ids"],
        batch["attention_mask"],
        batch["labels"],
    )
    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=label_ids)
    return outputs[1].cpu().detach().numpy()


def predictor(model, ds, instance, texts):
    instances = []
    for text in texts:
        inst_copy = copy(instance)
        inst_copy["source"] = text.replace("UNKWORDZ", "[MASK]")
        instances.append(ds.process_item(inst_copy))

    predns = []
    for batch in lazy_groups_of(instances, 64):
        predns.append(model_predict(model, move(ds.collate_fn(batch), "cuda")))

    return np.row_stack(predns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    reader = FEVERReader(True)
    generator = reader.read(args.in_file)
    ds = FEVERClsDataset(tokenizer, [], 256)

    with torch.no_grad(), open(args.out_file, "w+") as log:
        model = AutoModelForSequenceClassification.from_pretrained(args.model).to(
            "cuda"
        )
        model.eval()
        explainer = LimeTextExplainer(bow=False)

        for instance in tqdm(
            filter(lambda i: i["label"] != "NOT ENOUGH INFO", generator)
        ):
            split_claim = " ".join(
                tokenizer.convert_tokens_to_string(
                    tokenizer.tokenize(instance["source"])
                ).split()
            )
            exp_h = explainer.explain_instance(
                split_claim,
                lambda texts: predictor(model, ds, instance, texts),
                num_features=6,
                top_labels=1,
                num_samples=250,
            )

            best_toks = list(exp_h.as_map().items())[0][1]
            best_keys = list(exp_h.as_map().keys())[0]
            prem_idx = [int(item[0]) for item in best_toks if item[1] >= 0]

            instance.update(
                {"original_claim": split_claim, "master_explanation": prem_idx}
            )

            log.write(json.dumps(instance, cls=NpEncoder) + "\n")
