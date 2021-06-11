import json
import random

from argparse import ArgumentParser
from transformers import AutoTokenizer


def all_lower(tokens):
    return [a.lower() for a in tokens]


def all_in(tokens, token_set):
    return [t in token_set for t in tokens]


def mask_uncommon(tokens, mask, mask_token="*"):
    return [t if m else mask_token for m, t in zip(mask, tokens)]


def common_tokens(sentence1, sentence2):
    common = set(all_lower(sentence1)).intersection(all_lower(sentence2))
    return mask_uncommon(
        sentence1, all_in(all_lower(sentence1), common)
    ), mask_uncommon(sentence2, all_in(all_lower(sentence2), common))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    args = parser.parse_args()

    random = random.Random()
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    with open(args.in_file) as f, open(args.out_file, "w+") as of:
        for line in f:
            instance = json.loads(line)

            tokens = tok.tokenize(instance["mutated"])
            words = tok.convert_tokens_to_string(tokens)
            ev_str = " ".join([" ".join(a) for a in instance["pipeline_text"]])
            ev_tokens = tok.tokenize(ev_str)
            ev_words = tok.convert_tokens_to_string(ev_tokens)

            original_mask, mutated_mask = common_tokens(words.split(), ev_words.split())
            explanation = [idx for idx, tok in enumerate(original_mask) if tok == "*"]

            instance["original_claim"] = words
            instance["master_explanation"] = explanation

            of.write(json.dumps(instance) + "\n")
