# sh ./NNConnectivitiy/scripts/nnc_cifar10.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


python -u main.py \
    --seed 2025 \
    --id_ 7 \
    --data_path "" \
    --log_path ./NNConnectivitiy/Logs_/logs_ \
    --gpu_dev 0 \
    --dataset "CIFAR-10" \
    --network "lenet" \
    --train_mode 2 \
    --equalize_init 1 \
    --epochs 300 \
    --batch 256 \
    --batch_testing 256 \
    --lr_ 0.0015 \
    --opti sgd

