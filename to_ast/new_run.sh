#!/bin/bash
set -e
set -x

# Modelos preentrenados & GPU
export TORCH_HOME=../../pretrained_models
export CUDA_VISIBLE_DEVICES=0

# Parámetros del experimento
model=ast
dataset=animal_sounds
imagenetpretrain=True
audiosetpretrain=True
bal=none

if [ "$audiosetpretrain" = "True" ]; then
  lr=1e-5
else
  lr=1e-4
fi

freqm=24
timem=96
mixup=0
epoch=25
batch_size=48
fstride=10
tstride=10

dataset_mean=-6.6268077
dataset_std=5.358466
audio_length=512
noise=False

metrics=acc
loss=CE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

base_exp_dir=./exp/test-${dataset}-f${fstride}-t${tstride}-imp${imagenetpretrain}-asp${audiosetpretrain}-b${batch_size}-lr${lr}


# Salir si ya existe
if [ -d "$base_exp_dir" ]; then
  echo "¡El experimento ya existe en ${base_exp_dir}! Abortando."
  exit 1
fi
mkdir -p "$base_exp_dir"

# Bucle de 5 folds
for fold in 1 2 3 4 5; do
  echo "=== Procesando fold ${fold} ==="

  exp_dir=${base_exp_dir}/fold${fold}
  mkdir -p "$exp_dir"

  tr_data=./data/datafiles/animal_sounds_train_data_${fold}.json
  te_data=./data/datafiles/animal_sounds_eval_data_${fold}.json

  CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py \
    --model              ${model} \
    --dataset            ${dataset} \
    --data-train         ${tr_data} \
    --data-val           ${te_data} \
    --exp-dir            ${exp_dir} \
    --label-csv          ./data/animal_sounds_class_labels_indices.csv \
    --n_class            43 \
    --lr                 ${lr} \
    --n-epochs           ${epoch} \
    --batch-size         ${batch_size} \
    --save_model         True \
    --freqm              ${freqm} \
    --timem              ${timem} \
    --mixup              ${mixup} \
    --bal                ${bal} \
    --fstride            ${fstride} \
    --tstride            ${tstride} \
    --imagenet_pretrain  ${imagenetpretrain} \
    --audioset_pretrain  ${audiosetpretrain} \
    --metrics            ${metrics} \
    --loss               ${loss} \
    --warmup             ${warmup} \
    --lrscheduler_start  ${lrscheduler_start} \
    --lrscheduler_step   ${lrscheduler_step} \
    --lrscheduler_decay  ${lrscheduler_decay} \
    --dataset_mean       ${dataset_mean} \
    --dataset_std        ${dataset_std} \
    --audio_length       ${audio_length} \
    --noise              ${noise}
done

# Resumen de resultados
python ./get_esc_result.py --exp_path ${base_exp_dir}
