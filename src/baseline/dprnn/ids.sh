#!/bin/sh

gpu_id=14,15

continue_from=''

if [ -z ${continue_from} ]; then
	log_name='dprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=7886 \
main.py \
--batch_size 4 \
\
--log_name $log_name \
\
--use_tensorboard 1 \
>logs/$log_name/console1.txt 2>&1

# --continue_from ${continue_from} \