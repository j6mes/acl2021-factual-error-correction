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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in_file")
    args = parser.parse_args()

    any_masks = 0
    total_read = 0
    mask_lens = []
    tok_lens = []
    mask_prop = []

    with open(args.in_file) as f:
        for line in f:
            instance = json.loads(line)
            total_read += 1

            tok_lens.append(len(instance["original_claim"].split()))
            if len(instance["master_explanation"]):
                any_masks += 1
                mask_lens.append(len(instance["master_explanation"]))
                mask_prop.append(mask_lens[-1] / tok_lens[-1])

        print(f"Read {total_read} instances, of which {any_masks} have masks")
        print("Average mask length is: ", sum(mask_lens) / len(mask_lens))
        print("Average mask prop is: ", sum(mask_prop) / len(mask_prop))
        print("Average tok length is: ", sum(tok_lens) / len(tok_lens))

        print()
