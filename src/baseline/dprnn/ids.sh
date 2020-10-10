#!/bin/sh

gpu_id=2,3


continue_from=

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
--master_port=1152 \
main.py \
\
--train_dir '/home/panzexu/workspace/speech_separation/data/tr' \
--valid_dir '/home/panzexu/workspace/speech_separation/data/cv' \
--test_dir '/home/panzexu/workspace/speech_separation/data/tt' \
\
--batch_size 32 \
\
--L 20 \
--K 100 \
\
--log_name $log_name \
\
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \