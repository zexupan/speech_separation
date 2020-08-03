#!/bin/sh
#SBATCH --gres=gpu:2
#SBATCH --output=hlt.txt
#SBATCH --nodelist=ttnusa3


log_name='Dprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
mkdir logs/$log_name

python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=5436 \
main.py \
\
--batch_size 4 \
\
--L 40 \
--K 100 \
--B 128 \
--audio_direc '/data_a8/zexu/datasets/LRS2/audio/Audio/' \
\
--mix_lst_path '/data_a8/zexu/datasets/LRS2/audio/3_mix_min/mixture_data_list_3mix.csv' \
--mixture_direc '/data_a8/zexu/datasets/LRS2/audio/3_mix_min/' \
--C 3 \
--epochs 100 \
\
--log_name $log_name \
\
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \


