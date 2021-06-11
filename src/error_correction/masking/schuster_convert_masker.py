import json
from argparse import ArgumentParser

if __name__ == "__main__":
    references = []
    predictions = []

    parser = ArgumentParser()
    parser.add_argument("prediction_file")
    parser.add_argument("out_file")
    args = parser.parse_args()

    with open(args.prediction_file) as inputs, open(args.out_file, "w+") as final:

        for line_in in inputs:
            instance = json.loads(line_in)

            final.write(
                json.dumps(
                    {
                        "master_explanation": instance["masked_inds"],
                        "original_claim": instance["sentence1"],
                        "original_evidence": instance["sentence2"],
                        "target": instance["target"],
                        "original_id": instance["original"]["original"]["original_id"],
                        "claim_id": instance["original"]["original"]["claim_id"],
                        "actual_label": instance["original"]["veracity"],
                    }
                )
                + "\n"
            )
