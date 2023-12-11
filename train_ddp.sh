export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export PATH=/usr/local/cuda/bin:$PATH


whisper_path=pretrained_models/whisper-small

DATA_ROOT=data/now_how2/
SAVE_ROOT=checkpoints/now_how2_v2/

mkdir -p $SAVE_ROOT

python -m torch.distributed.run --nproc_per_node=3 --master_port=52011 train.py \
    --deepspeed config/dp_config_zero1.json \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "*.jsonl" \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  True \
    \
    --learning_rate 1e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    \
    --whisper_model $whisper_path \
    \
    --disable_tqdm True \
    \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 1 \
    --overwrite_output_dir