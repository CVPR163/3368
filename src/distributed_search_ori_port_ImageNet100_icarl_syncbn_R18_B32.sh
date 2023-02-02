#!/bin/bash
NUM_PROC=$1
NUM_NODES=$2
NODE_RANK=$3
host=$4
port=$5
l_alpha=$6
l_beta=$7
l_gamma=$8
num_tasks=$9
appr=icarl
bz=32
buffer_size=2000
datasets=imagenet_100
exemplar=herding
lamb=1.0
lr=0.1
minibatch_size_1=32
minibatch_size_2=32
mom=0.9
nc_first_task=50
nepochs=90
network=resnet18
num_exemplars_per_class=20
seed=0
weight_decay=1e-4


echo $@
export PYTHONWARNINGS="ignore"
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK --master_addr=$host --master_port=$port \
    main_incremental.py --l-alpha $l_alpha --l-beta $l_beta --l-gamma $l_gamma \
    --num-tasks $num_tasks --approach $appr --batch-size $bz --buffer-size $buffer_size \
    --datasets $datasets --exemplar-selection $exemplar --lamb $lamb --lr $lr \
    --minibatch-size-1 $minibatch_size_1 --minibatch-size-2 $minibatch_size_2 \
    --momentum $mom --nc-first-task $nc_first_task --nepochs $nepochs --network $network \
    --num-exemplars-per-class $num_exemplars_per_class --seed $seed --weight-decay $weight_decay \
    --save-models --ddp --syncbn "$@"