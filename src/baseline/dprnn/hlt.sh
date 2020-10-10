#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --output=hlt.txt
#SBATCH --nodelist=ttnusa4

continue_from='Dprnn_07-10-2020(00:35:11)'

if [ -z ${continue_from} ]; then
	log_name='Dprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=5466 \
main.py \
\
--train_dir '/home/zexu/workspace/speech_separation/data/tr' \
--valid_dir '/home/zexu/workspace/speech_separation/data/cv' \
--test_dir '/home/zexu/workspace/speech_separation/data/tt' \
\
--batch_size 32 \
\
--L 20 \
--K 100 \
\
--continue_from ${continue_from} \
--log_name $log_name \
--use_tensorboard 1 \
>logs/$log_name/console1.txt 2>&1

# --continue_from ${continue_from} \

# --opt-level O0 \
# --use_tensorboard 1 \
