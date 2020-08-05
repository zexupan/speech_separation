#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --output=hlt.txt
#SBATCH --nodelist=ttnusa3


continue_from='Conv_04-08-2020(16:29:35)'

if [ -z ${continue_from} ]; then
	log_name='Conv_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=5586 \
main.py \
\
--train_dir '/data07/zexu/workspace/speech_separation/data/tr' \
--valid_dir '/data07/zexu/workspace/speech_separation/data/cv' \
--test_dir '/data07/zexu/workspace/speech_separation/data/tt' \
\
--log_name $log_name \
--continue_from ${continue_from} \
\
--use_tensorboard 1 \
>logs/$log_name/console1.txt 2>&1




