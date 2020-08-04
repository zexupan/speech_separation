#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --output=hlt.txt
#SBATCH --nodelist=ttnusa3


log_name='Conv_'$(date '+%d-%m-%Y(%H:%M:%S)')
mkdir logs/$log_name

python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=5556 \
main.py \
\
--train_dir '/data07/zexu/workspace/speech_separation/data/tr' \
--valid_dir '/data07/zexu/workspace/speech_separation/data/cv' \
--test_dir '/data07/zexu/workspace/speech_separation/data/tt' \
\
--log_name $log_name \
\
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \


