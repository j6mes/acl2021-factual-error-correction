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
from operator import itemgetter

from argparse import ArgumentParser
from drqa.retriever.utils import normalize
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dpr_file")
    parser.add_argument("genre_file")
    parser.add_argument("output_file")
    parser.add_argument("--num_ctx", type=int, default=2)

    args = parser.parse_args()

    with open(args.dpr_file) as dpr_f, open(args.genre_file) as genre_f, open(
        args.output_file, "w+"
    ) as out_f:

        for idx, (dpr_line, genre_line) in tqdm(enumerate(zip(dpr_f, genre_f))):

            dpr_instance = json.loads(dpr_line)
            genre_instance = json.loads(genre_line)

            assert dpr_instance["id"] == genre_instance["id"], idx

            genre_predictions = [
                normalize(p["title"])
                for p in sorted(
                    genre_instance["output"][0]["provenance"],
                    key=itemgetter("score"),
                    reverse=True,
                )
            ]
            metadata = {
                "genre_pipeline": genre_instance["output"],
                "genre_pages": genre_predictions,
            }
            genre_predictions = set(genre_predictions)
            metadata["matches"] = [
                a
                for a in dpr_instance["pipeline_text"]
                if normalize(a[0]) in genre_predictions
            ]
            dpr_instance["genre"] = metadata
            if len(metadata["matches"]):
                dpr_instance["pipeline_text"] = metadata["matches"][: args.num_ctx]
            else:
                dpr_instance["pipeline_text"] = dpr_instance["pipeline_text"][
                    : args.num_ctx
                ]

            out_f.write(json.dumps(dpr_instance) + "\n")
