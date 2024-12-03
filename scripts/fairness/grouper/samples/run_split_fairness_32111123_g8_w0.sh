
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="invariant/datasets/fairness/split_fairness_32111123"

echo "start split_fairness_32111123 samples"
accelerate launch --mixed_precision="fp16"  src/group/generate_train_text_to_image.py \
  --path_prefix=/your/path/to/this/repo/InvDiff/data/ \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="invariant/ckpts/fairness/groupers/samples/split_fairness_32111123/" \
  --based_unet_path="invariant/ckpts/fairness/models/split_fairness_32111123-noeiil-nodelta-lambda0-delta0-small-deltaparam0-False-4-t_g2_w3-bs64-sc1e-04-ac1-lrcosine-wu1000/checkpoint-10000" 

echo "finish split_fairness_32111123 samples"


echo "start split_fairness_32111123 groupers group_num=8 loss_not_zero_weight=0"
python src/group/train_group.py     --path_embedding_file="invariant/ckpts/fairness/groupers/samples/split_fairness_32111123/samples_t.pkl"     --path_prefix="/your/path/to/this/repo/InvDiff/data/"     --path_groupers="invariant/ckpts/fairness/groupers/groupers/split_fairness_32111123"     --group_num=8     --loss_not_zero_weight=0 
echo "finish split_fairness_32111123 groupers"
