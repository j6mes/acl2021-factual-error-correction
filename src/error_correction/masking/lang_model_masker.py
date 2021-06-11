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
import heapq
import json
from argparse import ArgumentParser

from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mlm_file")
    parser.add_argument("out_file")
    parser.add_argument("--top_k", type=int, required=True)
    args = parser.parse_args()

    k = args.top_k

    def deduplicate(evidence):
        unique = set(map(lambda ev: (ev["page"], ev["line"]), evidence))
        return unique

    with open(args.mlm_file) as infile, open(args.out_file, "w+") as outfile:
        for line in tqdm(infile):
            instance = json.loads(line)
            top = heapq.nlargest(
                k, range(len(instance["scores"])), instance["scores"].__getitem__
            )

            masked = " ".join([tok if idx not in top else ("[MASK]" if "##" not in tok else "##[MASK]") for idx,tok in enumerate(instance["mask_tokens"])]).replace(" ##","")
            idxs = [idx for idx,val in enumerate(masked.split()) if "[MASK]" in val]

            instance["original_claim"] = " ".join(instance["mask_tokens"]).replace(
                " ##", ""
            )
            instance["master_explanation"] = idxs

            del instance["scores"]
            del instance["mask_tokens"]
            outfile.write(json.dumps(instance) + "\n")
