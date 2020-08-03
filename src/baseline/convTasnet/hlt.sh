#!/bin/sh
#SBATCH --gres=gpu:2
#SBATCH --output=hlt.txt
#SBATCH --nodelist=ttnusa3

gpu_id=2,3

log_name='Conv_'$(date '+%d-%m-%Y(%H:%M:%S)')
# mkdir logs/$log_name

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=5556 \
main.py \
\
--train_dir '/data_a8/zexu/workspace/speech_separation/data/tr' \
--valid_dir '/data_a8/zexu/workspace/speech_separation/data/cv' \
--test_dir '/data_a8/zexu/workspace/speech_separation/data/tt' \
\
--log_name $log_name \
\
# --use_tensorboard 1 \
# >logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \


