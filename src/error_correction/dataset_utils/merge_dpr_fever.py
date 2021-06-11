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
import json
from tqdm import tqdm


def stream_read_json(f):
    start_pos = 0
    while True:
        try:
            obj = json.load(f)
            yield obj
            return
        except json.JSONDecodeError as e:
            f.seek(start_pos)
            json_str = f.read(e.pos)
            obj = json.loads(json_str)
            start_pos += e.pos
            yield obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fever_file")
    parser.add_argument("dpr_file")
    parser.add_argument("out_file")
    parser.add_argument("--num_ctx", type=int, default=1)
    args = parser.parse_args()

    with open(args.fever_file) as fever, open(args.dpr_file) as dpr, open(
        args.out_file, "w+"
    ) as out_file:
        dpr_results = json.load(dpr)
        dpr_ptr = 0
        for idx, line in tqdm(enumerate(fever)):
            instance = json.loads(line)
            dpr_result = dpr_results[dpr_ptr]

            assert (
                dpr_result["question"].replace('"', "")
                == instance["claim"].replace('"', "")
                or dpr_result["question"] == instance["claim"]
            ), f"Claim missmatch on line {idx} (sync to {dpr_ptr}) of file {args.fec_file}\n\n{line}"

            instance["pipeline_text"] = []
            for context in dpr_result["ctxs"][: args.num_ctx]:
                instance["pipeline_text"].append((context["title"], context["text"]))

            out_file.write(json.dumps(instance) + "\n")
            dpr_ptr += 1
