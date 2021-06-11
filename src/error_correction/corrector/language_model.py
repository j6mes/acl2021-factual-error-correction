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

from argparse import ArgumentParser
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
import torch

from error_correction.modelling.reader.mask_based_correction_reader import MaskBasedCorrectionReader


def move(dict_of_tensors, device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in dict_of_tensors.items()}


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to("cuda")
    reader = MaskBasedCorrectionReader({"SUPPORTS","REFUTES"}, True)
    parser = ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()


    with open(args.output_file,"w+") as of, torch.no_grad():

        for line in tqdm(reader.read(args.input_file)):

            line["prediction"] = line["source"].replace(" ##", "").strip()
            inputs = tokenizer(line["source"], return_tensors='pt')
            original_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
            original_string = tokenizer.convert_tokens_to_string(original_tokens)

            line["tokenized_input"] = original_string

            while "[MASK]" in line["prediction"]:

                inputs = tokenizer(line["prediction"], return_tensors='pt')
                original_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                original_string = tokenizer.convert_tokens_to_string(original_tokens)

                mask_positions = [idx for idx,a in enumerate(original_string.split()) if a == "[MASK]"]

                outputs = model(**move(inputs, "cuda"))
                predictions = outputs[0].cpu()

                prediction_ids = predictions[0].argmax(dim=1)
                p_ids = [(idx, prediction_ids[idx].data.item()) for idx in mask_positions]

                ps = []

                for sentence_pos, tok_num in p_ids:
                    ps.append(predictions[0,sentence_pos,tok_num])

                best_pos = torch.tensor(ps).argmax(dim=0).data.item()
                best_tok = tokenizer.convert_ids_to_tokens([p_ids[best_pos][1]])[0]

                returned_toks = " ".join([a if idx != p_ids[best_pos][0] else best_tok for idx,a in enumerate(original_string.split())])
                line["prediction"] = returned_toks.replace("[CLS]","").replace("[SEP]","").strip().replace(" ##", "").strip()
            print(line["prediction"])
            of.write(json.dumps(line)+"\n")