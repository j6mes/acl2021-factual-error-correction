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
from argparse import ArgumentParser

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("mlm_file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    mid = tokenizer.mask_token_id

    with open(args.in_file) as f, open(args.out_file, "w+") as of:
        for line in tqdm(f):
            instance = json.loads(line)
            instance["scores"] = []
            instance["mask_tokens"] = []
            inputs = tokenizer(instance["mutated"], return_tensors="pt", padding=True)
            inputs["labels"] = inputs["input_ids"].clone()

            # input is [CLS] X X X [SEP], so skip the first and last tokens
            for i in range(inputs["input_ids"].shape[1] - 2):
                inputs["input_ids"] = inputs["labels"].clone()
                inputs["input_ids"][0][1 + i] = mid
                tok_id = inputs["labels"][0][1 + i]

                outputs = model(**inputs)

                instance["scores"].append(outputs[0].item())
                instance["mask_tokens"].append(tokenizer.decode([tok_id]))
            of.write(json.dumps(instance) + "\n")
