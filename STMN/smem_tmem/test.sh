TRAIN_TXT=../MEVID_database/train_path.txt
TRAIN_INFO=../MEVID_database/train_info.npy
TEST_TXT=../MEVID_database/test_path.txt
TEST_INFO=../MEVID_database/test_info.npy
QUERY_INFO=../MEVID_database/query_IDX.npy

CKPT=./log
LOAD_CKPT=log/STMN_best_model.pth

python3 main.py \
    --eval_only \
    --smem_size 10 --smem_margin 0.3 \
    --tmem_size 5 --tmem_margin 0.3 \
    --train_txt $TRAIN_TXT --train_info $TRAIN_INFO  --test_batch 1 \
    --test_txt $TEST_TXT  --test_info $TEST_INFO --query_info $QUERY_INFO \
    --ckpt $CKPT --load_ckpt $LOAD_CKPT --pin_memory False \
    --class_per_batch 6 --track_per_class 4 --seq_len 6 \
    --feat_dim 2048 --stride 1 --gpu_id '1'
