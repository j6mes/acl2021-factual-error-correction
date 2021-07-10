lr=$1
reader=$2
ns=$3
dpr=$4

python -m error_correction.classifier.run \
  --model_name_or_path bert-base-uncased \
  --learning_rate $lr \
  --num_train_epochs 3 \
  --output_dir output/fever/reader=${reader},lr=${lr},data=${dpr},negative_sampling=${ns}/seed=1/  \
  --train_file resources/retrieval/genre_50/${dpr}_train.jsonl \
  --val_file resources/retrieval/genre_50/${dpr}_dev.jsonl \
  --do_train \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --reader $reader \
  --negative_sampling ${ns} \
  --seed 1
