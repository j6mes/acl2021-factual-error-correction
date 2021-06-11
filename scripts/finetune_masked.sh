model=$1
lr=$2
masker=$3
mutation_source=false
mutation_target=false
labels=$4
bs=${5:-8}
seed=${SEED:-42}

output_dir=output/masker/model=${model},lr=${lr},masker=${masker},mutation_source=${mutation_source},mutation_target=${mutation_target},labels=${labels}/seed=$seed/

python \
  -m error_correction.corrector.run \
  --model_name_or_path $model \
  --output_dir $output_dir \
  --train_file resources/masking/${masker}_train.jsonl \
  --val_file resources/masking/${masker}_dev.jsonl \
  --do_train \
  --train_batch_size $bs \
  --learning_rate $lr \
  --seed $seed \
  --num_train_epochs 8 \
  --reader mask \
  --mutation_source ${mutation_source} \
  --mutation_target ${mutation_target} \
  --labels ${labels}