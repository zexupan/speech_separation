#!/bin/sh

gpu_id=13,14

continue_from=''

if [ -z ${continue_from} ]; then
	log_name='Dprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
	# mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=1553 \
main.py \
\
--opt-level O0 \
\
--batch_size 2 \
--mix_lst_path '/home/panzexu/datasets/LRS2/audio/2_mix_min/mixture_data_list_2mix.csv' \
--mixture_direc '/home/panzexu/datasets/LRS2/audio/2_mix_min/' \
--C 2 \
\
--log_name $log_name \
\
# --use_tensorboard 1 \
# >logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \

# --mix_lst_path '/home/panzexu/datasets/LRS2/audio/3_mix_min/mixture_data_list_3mix.csv' \
# --mixture_direc '/home/panzexu/datasets/LRS2/audio/3_mix_min/' \
# --C 3 \