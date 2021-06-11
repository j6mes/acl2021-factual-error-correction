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
