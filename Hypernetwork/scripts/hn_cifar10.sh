# sh ./Hypernetwork/scripts/hn_cifar10.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


python -u main_hn.py \
    --seed 2025 \
    --id_ 7 \
    --data_path /data/home/mkim332/data/CV_data \
    --log_path ./Hypernetwork/Logs_/logs_ \
    --gpu_dev 0 \
    --dataset "CIFAR-10" \
    --network "lenet" \
    --init_hyper 1 \
    --train_mode 0 \
    --equalize_init 0 \
    --epochs 300 \
    --epochs_hn 2000 \
    --batch 256 \
    --batch_testing 256 \
    --lr_ 0.0015 \
    --opti sgd

