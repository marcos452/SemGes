echo "[Test] SpeechCLIP Cascaded plus on Flickr8k"
EXP_ROOT="exp_test"
DATASET_ROOT="/mnt/md1/user_jeffwang/Dataset/flickr"
mkdir $EXP_ROOT

python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume 'path/to/your/ckpt' \
    --dataset_root ${DATASET_ROOT} \
    --gpus 2 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT