# sh ./gradientNNs/scripts/ETTh1.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=1
data_path_name=""
data=ETTm1

look_back=128
freq_span=-1
random_seed=88
run_id=333
# batch_size=442
batch_size=268
logpath='./gradientNNs/Logs_/logs_'
for (( i=0; i<${runs}; i++ ));
do
    pred_len=128
    python -u grad_main.py \
      --log_path $logpath \
      --seed $random_seed \
      --data_path $data_path_name \
      --id_ $run_id \
      --dataset $data \
      --look_back $look_back \
      --horizon $pred_len \
      --vars_in_train $look_back $look_back $pred_len $look_back \
      --vars_in_test $look_back $look_back $pred_len $look_back \
      --network tsmixer \
      --channel_dependence 1 \
      --input_c 7 \
      --hidden_dim 36 \
      --n_epochs 80 \
      --scheduler 0 \
      --warm_up 0.2 \
      --final_lr 0.00005\
      --ref_lr 0.00015 \
      --start_lr 0.00005 \
      --description "" \
      --gpu_dev 1 \
      --patience 6 \
      --batch $batch_size --batch_testing 64 --lr_ 0.0001
done