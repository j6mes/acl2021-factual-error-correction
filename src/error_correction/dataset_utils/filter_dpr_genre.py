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
