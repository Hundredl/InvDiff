
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="invariant/datasets/fairness/split_fairness_32111123"
export PATH_PREFIX=/your/path/to/this/repo/InvDiff/data/


export LAMBDA=0
export DELTA_TYPE="small"
export SOFT_GROUPER=False
export PATH_GROUPER=t_g2_w3 # "t_g2_w3"
export LEARNING_RATE=1e-04
export BATCH_SIZE=64
export AC_STEPS=1
export LR_SCHEDULER="cosine"
export WARMUP_STEPS=1000
export DELTA_PARAM=0
export DELTA_RATIO=0
export HARD_GROUPER_NUM=4

accelerate launch --mixed_precision="fp16" src/train_text_to_image.py \
  --path_prefix=$PATH_PREFIX \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --dataset="fairness" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=${BATCH_SIZE} \
  --gradient_accumulation_steps=${AC_STEPS} \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --learning_rate=${LEARNING_RATE} \
  --max_grad_norm=1 \
  --lr_scheduler=${LR_SCHEDULER} --lr_warmup_steps=${WARMUP_STEPS} \
  --checkpointing_steps=1000 \
  --eiil=0 \
  --delta=False \
  --lambda_value=${LAMBDA} \
  --delta_ratio=${DELTA_RATIO} \
  --delta_type=${DELTA_TYPE} \
  --delta_param=${DELTA_PARAM} \
  --wandb=True \
  --soft_grouper=${SOFT_GROUPER} \
  --hard_grouper_num=${HARD_GROUPER_NUM} \
  --output_dir="invariant/ckpts/fairness/models/split_fairness_32111123-noeiil-nodelta-lambda${LAMBDA}-delta${DELTA_RATIO}-${DELTA_TYPE}-deltaparam${DELTA_PARAM}-${SOFT_GROUPER}-${HARD_GROUPER_NUM}-${PATH_GROUPER}-bs${BATCH_SIZE}-sc${LEARNING_RATE}-ac${AC_STEPS}-lr${LR_SCHEDULER}-wu${WARMUP_STEPS}" \
