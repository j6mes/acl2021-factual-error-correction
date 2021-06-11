import heapq
import json
from argparse import ArgumentParser

from tqdm import tqdm

from error_correction.modelling.reader.fever_database import FEVERDocumentDatabase

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
