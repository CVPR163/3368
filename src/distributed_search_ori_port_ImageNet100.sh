#!/bin/bash
NUM_PROC=$1
NUM_NODES=$2
NODE_RANK=$3
host=$4
port=$5

appr=lucir
bz=128
buffer_size=2000
clipping=-1
datasets=imagenet_100
exemplar=herding
first_task_bz=128
first_task_lr=0.1

l_alpha=$6
l_beta=$7
l_gamma=$8

lamb=10.0
lr=0.1
minibatch_size_1=16
minibatch_size_2=128
mom=0.9
nc_first_task=50
nepochs=90
network=resnet18
num_exemplars_per_class=0

num_tasks=$9

seed=0
weight_decay=1e-4


echo $@
export PYTHONWARNINGS="ignore"
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK --master_addr=$host --master_port=$port \
    main_incremental.py --approach $appr --batch-size $bz --buffer-size $buffer_size \
    --clipping $clipping --datasets $datasets --exemplar-selection $exemplar --first-task-bz $first_task_bz \
    --first-task-lr $first_task_lr --l-alpha $l_alpha --l-beta $l_beta --l-gamma $l_gamma --lamb $lamb \
    --lr $lr --minibatch-size-1 $minibatch_size_1 --minibatch-size-2 $minibatch_size_2 --momentum $mom \
    --nc-first-task $nc_first_task --nepochs $nepochs --network $network --num-exemplars-per-class $num_exemplars_per_class \
    --num-tasks $num_tasks --seed $seed --weight-decay $weight_decay --save-models --ddp "$@"