# sh ./Hypernetwork/scripts/hn_mnist.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


python -u main_hn.py \
    --seed 2025 \
    --id_ 14 \
    --data_path /data/home/mkim332/data/CV_data \
    --log_path ./Hypernetwork/Logs_/logs_ \
    --gpu_dev 0 \
    --dataset "MNIST" \
    --network "mlp" \
    --init_hyper 1 \
    --hypernet_training reconstruction \
    --equalize_init 1 \
    --lowrank 1 \
    --decomposition 0 \
    --hypermatching 0 \
    --gradient_matching 0 \
    --multitask 1 \
    --epochs 34 \
    --epochs_hn 30000 \
    --batch 256 \
    --batch_testing 256 \
    --lr_ 0.001 \
    --opti adam

# python -u main_hn.py \
#     --seed 2025 \
#     --id_ 14 \
#     --data_path /data/home/mkim332/data/CV_data \
#     --log_path ./Hypernetwork/Logs_/logs_ \
#     --gpu_dev 1 \
#     --dataset "MNIST" \
#     --network "lenet" \
#     --init_hyper 1 \
#     --hypernet_training reconstruction \
#     --equalize_init 0 \
#     --lowrank 1 \
#     --decomposition 0 \
#     --hypermatching 0 \
#     --epochs 34 \
#     --epochs_hn 40000 \
#     --batch 256 \
#     --batch_testing 256 \
#     --lr_ 0.001 \
#     --opti adam

