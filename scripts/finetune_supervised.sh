model=$1
lr=$2
mutation_source=$3
mutation_target=$4
labels=$5
bs=${6:-8}
seed=${SEED:-42}

output_dir=output/supervised/model=${model},reader=${reader},lr=${lr},mutation_source=${mutation_source},mutation_target=${mutation_target},labels=${labels}/seed=$seed/

python \
  -m error_correction.corrector.run \
  --model_name_or_path $model \
  --output_dir $output_dir \
  --train_file resources/mutation/train.jsonl \
  --val_file resources/mutation/dev.jsonl \
  --do_train \
  --train_batch_size $bs \
  --learning_rate $lr \
  --seed $seed \
  --num_train_epochs 8 \
  --mutation_source ${mutation_source} \
  --mutation_target ${mutation_target} \
  --labels ${labels} \
  --reader supervised