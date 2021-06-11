import argparse
import json
import math
import random

from tqdm import tqdm
from transformers import AutoTokenizer
from error_correction.masking.json_encoder import NpEncoder
from error_correction.modelling.reader.fever_reader import FEVERReader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("--sample_prob", type=float, default=0.15)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    reader = FEVERReader(True)
    generator = reader.read(args.in_file)

    with open(args.out_file, "w+") as log:
        for instance in tqdm(
            filter(lambda i: i["label"] != "NOT ENOUGH INFO", generator)
        ):
            split_claim = tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(instance["source"])
            ).split()
            prem_idx = random.sample(
                range(len(split_claim)),
                k=max(
                    1,
                    min(
                        len(split_claim), math.ceil(len(split_claim) * args.sample_prob)
                    ),
                ),
            )

            instance.update(
                {"original_claim": " ".join(split_claim), "input_mask": prem_idx}
            )
            log.write(
                json.dumps(
                    {
                        "instance": instance,
                    },
                    cls=NpEncoder,
                )
                + "\n"
            )
