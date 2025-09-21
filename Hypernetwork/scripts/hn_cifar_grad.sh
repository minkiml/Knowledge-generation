# sh ./Hypernetwork/scripts/hn_cifar_grad.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


python -u main_hn_grad.py \
    --seed 2025 \
    --id_ 5 \
    --data_path "" \
    --log_path ./Hypernetwork/Logs_/logs_ \
    --gpu_dev 0 \
    --dataset "CIFAR-10" \
    --network "resnet" \
    --init_hyper 1 \
    --hypernet_training prediction \
    --equalize_init 1 \
    --lowrank 1 \
    --decomposition 0 \
    --hypermatching 0 \
    --gradient_matching 0 \
    --intrinsic_training 0\
    --grad_training 1 \
    --h 2 \
    --multitask 0 \
    --epochs 34 \
    --epochs_hn 50 \
    --batch 64 \
    --batch_testing 256 \
    --lr_ 0.0008 \
    --opti adam
