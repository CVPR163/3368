mdevice_id=0
appr=lucir_wo_reservoir
bz=128
clipping=-1
datasets=cifar100_icarl
exemplar=herding
first_task_bz=128
first_task_lr=0.1
lamb=5.0
lr=0.1
mom=0.9
l_alpha=8.0
l_beta=1.0
l_gamma=0.25
nc_first_task=50
nepochs=160
network=resnet18_cifar
num_exemplars_per_class=20
seed=0
weight_decay=5e-4
num_tasks=6

CUDA_VISIBLE_DEVICES=$device_id python main_incremental.py --l-alpha $l_alpha --l-beta $l_beta --l-gamma $l_gamma \
    --num-tasks $num_tasks --approach $appr --batch-size $bz \
    --clipping $clipping --datasets $datasets --exemplar-selection $exemplar \
    --first-task-bz $first_task_bz --first-task-lr $first_task_lr --lamb $lamb \
    --lr $lr --momentum $mom \
    --nc-first-task $nc_first_task --nepochs $nepochs --network $network --num-exemplars-per-class $num_exemplars_per_class \
    --seed $seed --weight-decay $weight_decay --save-models