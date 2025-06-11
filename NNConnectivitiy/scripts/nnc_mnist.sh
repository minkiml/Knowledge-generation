# sh ./NNConnectivitiy/scripts/nnc_mnist.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


python -u main.py \
    --seed 2025 \
    --id_ 12 \
    --data_path /data/home/mkim332/data/CV_data \
    --log_path ./NNConnectivitiy/Logs_/logs_ \
    --gpu_dev 0 \
    --dataset "MNIST" \
    --network "lenet" \
    --equalize_init 1 \
    --epochs 60 \
    --batch 256 \
    --batch_testing 256 \
    --lr_ 0.002 \
    --opti sgd

