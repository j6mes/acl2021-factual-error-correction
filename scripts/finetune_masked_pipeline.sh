model=$1
lr=$2
masker=$3
mutation_source=false
mutation_target=false
labels=$4
data=$5
bs=${6:-4}
seed=${SEED:-42}

output_dir=output/masker/model=${model},lr=${lr},masker=${masker},data=${data},mutation_source=${mutation_source},mutation_target=${mutation_target},labels=${labels}/seed=$seed/

python \
  -m error_correction.corrector.run \
  --model_name_or_path $model \
  --output_dir $output_dir \
  --train_file resources/masking/${masker}_train_${data}.jsonl \
  --val_file resources/masking/${masker}_dev_${data}.jsonl \
  --do_train \
  --train_batch_size $bs \
  --learning_rate $lr \
  --seed $seed \
  --num_train_epochs 4 \
  --reader mask \
  --mutation_source ${mutation_source} \
  --mutation_target ${mutation_target} \
  --labels ${labels}