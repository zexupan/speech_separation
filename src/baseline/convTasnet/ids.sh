#!/bin/sh

gpu_id=14,13,12,8

continue_from='Conv_31-07-2020(14:29:53)'

if [ -z ${continue_from} ]; then
	log_name='Conv_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=7006 \
main.py \
\
--mix_lst_path '/home/panzexu/datasets/LRS2/audio/3_mix_min/mixture_data_list_3mix.csv' \
--mixture_direc '/home/panzexu/datasets/LRS2/audio/3_mix_min/' \
--C 3 \
\
--continue_from ${continue_from} \
--epochs 100 \
\
--log_name $log_name \
\
--use_tensorboard 1 \
>logs/$log_name/console1.txt 2>&1




# --mix_lst_path '/home/panzexu/datasets/LRS2/audio/3_mix_min/mixture_data_list_3mix.csv' \
# --mixture_direc '/home/panzexu/datasets/LRS2/audio/3_mix_min/' \
# --C 3 \