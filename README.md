# About
This is the code release to support the following paper, accepted at ACL2021: [Evidence-based Factual Error Correction](https://export.arxiv.org/pdf/2106.01072) by James Thorne and Andreas Vlachos, University of Cambridge.

## Data
Can be downloaded from this [Google Drive](https://drive.google.com/open?id=1hzwg5NtVUB_cfXiADkSanCq0JjaQ87tV) folder. Thank you for Nicola de Cao for generating the GENRE predictions, they have been merged with the predictions from DPR using the code release from this [Github Repo](https://github.com/facebookresearch/DPR). The merge script is in `src/error_correction/dataset_utils/filter_dpr_genre.py`.

## Environment
Create a new conda repository and `pip install -r requirements.txt`. 
All experiments expect `src` to be in the pythonpath:

`export PYTHONPATH=src`

## License
The source code released in this repository is released under the Apache 2.0 license. Some portions of the training script are
adapted from [examples released by HuggingFace](https://github.com/huggingface/transformers/tree/master/examples/pytorch), which is also released under an Apache 2.0 license and modified to support the error correction task.

The annotations for the FEVER dataset are released under the Creative Commons Share Alike License v3.0. More information can be found here [https://fever.ai/dataset/fever.html](https://fever.ai/dataset/fever.html)

# Running Experiments
## Masker
For all maskers, they can be used with either the gold evidence from the master file, or if `pipeline_text` is
present in the JSON document for each instance, the predicted evidence will be used for generating masks.  


### LIME-based masker (black-box)
Requires a pre-trained HF model that is trained as a (claim, evidence) pair classifier. This is quite slow though, 
it would be best to break the input file into multiple chunks or set it running and come back a day later 
(esp. when generating predictions for training dataset).

To first train the FEVER task classifier that will be used to generate lime predictions:

```
bash scripts/finetune_fever.sh 5e-5 fever_sub genre_50_2 true 
```

Then:

```
python -m error_correction.masking.lime_masker <in_file> <out_file> --model <path/to/huggingface/classifier>
```

### Random masker
Optional argument `--sample_prob=0.15`, defines the number of tokens masked from the claim.
```
python -m error_correction.masking.random_masker <in_file> <out_file>
```


### Masked-Language-Model masker
Required argument `--top_k=3`, defines the number of tokens to be masked (k tokens from MLM with highest per-token loss).

First use MLM to generate per-token mask predictions and save losses to mlm_file

```
python -m error_correction.masking.generate_mlm_losses <in_file> <mlm_file>
```

Then select the top k tokens from this file to use as an explanation
```
python -m error_correction.masking.lang_model_masker <mlm_file> <out_file> --top_k=3
```

### ESIM-based masker (white-box)
In a separate folder/conda environment, follow the instructions from this [GitHub repository](https://github.com/TalSchuster/TokenMasker) and run the `model_predictions.py` file on appropriate data split.
Several changes have been made to this repository, apply the patch in `third_party_patches/` from the `TokenMasker` repository. `git apply ../path/to/third_party_patches/schuster.patch`. 
There is an additional model_predictions file for using retrieved evidence instead of gold evidence called `model_predictions_pipeline.py`.

Then convert the predictions with this script select the top k tokens from this file to use as an explanation
```
python -m error_correction.masking.schuster_convert_masker <predictions_file> <out_file> 
```

## Corrector

### Fully-supervised ceiling

With gold evidence
```
bash scripts/finetune_supervised.sh t5-base 1e-4 true false all
```

With predicted evidence (from genre_dpr)
```
bash scripts/finetune_supervised_pipeline.sh t5-base 1e-4 true false all genre_dpr
```

To make predictions:

```
python -m error_correction.corrector.run \
    --model_name_or_path t5-base \
    --output_dir <path_to_trained_model_directory> \
    --do_predict \
    --test_file <path_to_test_file> \
    --reader supervised \
    --mutation_source true \
    --mutation_target false \
    --labels all
```


### Masker-corrector

Expects masked instances to be stored in `resources/masker/{masker_name}_{split}.jsonl` for gold evidence and
`resources/masker/{masker_name}_{split}_{ir_method}.jsonl` for evidence selected from IR system

With gold evidence
```
bash scripts/finetune_masked.sh t5-base 1e-4 {masker_name} true false all
```

With predicted evidence (from genre_dpr)
```
bash scripts/finetune_masked_pipeline.sh t5-base 1e-4 {masker_name} true false all {ir_method}
```

### Testing
