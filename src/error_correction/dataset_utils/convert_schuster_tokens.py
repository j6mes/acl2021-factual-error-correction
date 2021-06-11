import json
from collections import Counter

if __name__ == "__main__":

    found = dict()

    with open(
        "/local/scratch/jt719/fec-tals/TokenMaskerGenre/predictions/genre_50_1_dev.jsonl"
    ) as f:

        for line in f:
            instance = json.loads(line)
            if instance["original"]["original"]["claim_id"] not in found:
                instance["explanations"] = []
                found[instance["original"]["original"]["claim_id"]] = instance

            instance = found[instance["original"]["original"]["claim_id"]]
            instance["explanations"].append(instance["masked_inds"])

    with open(
        "resources/masking/merged_newschuster_genre_99_dev_genre_50_1.jsonl", "w+"
    ) as outf:
        for i in found.keys():
            instance = found[i]

            cnt = Counter()
            for exp in instance["explanations"]:
                if len(exp):
                    cnt[frozenset(exp)] += 1
            master = cnt.most_common(1)

            instance["master_explanation"] = []
            if len(master):
                instance["master_explanation"].extend(list(master[0][0]))

            instance["verdict"] = (
                None
                if instance["gold_label"] == "NOT ENOUGH INFO"
                else instance["gold_label"]
            )
            instance["original_claim"] = instance["sentence1"]
            instance["target"] = instance["original"]["original"]["original"]
            instance["master_explanation"] = instance["masked_inds"]
            instance["mutation"] = instance["original"]["original"]["mutation"]
            instance["pipeline_text"] = instance["original"]["original"][
                "pipeline_text"
            ]

            outf.write(json.dumps(instance) + "\n")
