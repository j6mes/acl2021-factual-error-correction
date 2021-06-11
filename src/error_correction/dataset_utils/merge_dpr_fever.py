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
