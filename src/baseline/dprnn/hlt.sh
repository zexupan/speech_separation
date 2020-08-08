#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --output=hlt.txt
#SBATCH --nodelist=ttnusa4

log_name='Dprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
mkdir logs/$log_name

python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=5556 \
main.py \
\
--batch_size 8 \
\
--train_dir '/data07/zexu/workspace/speech_separation/data/tr' \
--valid_dir '/data07/zexu/workspace/speech_separation/data/cv' \
--test_dir '/data07/zexu/workspace/speech_separation/data/tt' \
\
--log_name $log_name \
\
--L 20 \
--K 100 \
--opt-level O0 \
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \


