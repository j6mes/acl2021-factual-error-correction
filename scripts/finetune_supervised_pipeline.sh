model=$1
lr=$2
mutation_source=$3
mutation_target=$4
labels=$5
data=$6
bs=${7:-4}
seed=${SEED:-42}

output_dir=output/supervised/model=${model},reader=${reader},data=${data},lr=${lr},mutation_source=${mutation_source},mutation_target=${mutation_target},labels=${labels}/seed=$seed/

python \
  -m error_correction.corrector.run \
  --model_name_or_path $model \
  --output_dir $output_dir \
  --reader supervised \
  --train_file resources/retrieval/genre_50/${data}_train.jsonl \
  --val_file resources/retrieval/genre_50/${data}_dev.jsonl \
  --do_train \
  --train_batch_size $bs \
  --learning_rate $lr \
  --seed $seed \
  --num_train_epochs 4 \
  --mutation_source ${mutation_source} \
  --mutation_target ${mutation_target} \
  --labels ${labels}