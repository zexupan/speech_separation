#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --output=hlt.txt
#SBATCH --nodelist=ttnusa3

continue_from=

if [ -z ${continue_from} ]; then
	log_name='Conv_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=5586 \
main.py \
\
--train_dir '/home/zexu/workspace/speech_separation/data/tr' \
--valid_dir '/home/zexu/workspace/speech_separation/data/cv' \
--test_dir '/home/zexu/workspace/speech_separation/data/tt' \
\
--batch_size 8 \
\
--log_name $log_name \
\
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1




# --continue_from ${continue_from} \