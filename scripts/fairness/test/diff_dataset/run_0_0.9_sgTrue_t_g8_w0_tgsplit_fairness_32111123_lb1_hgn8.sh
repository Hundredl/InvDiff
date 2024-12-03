
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="invariant/datasets/fairness/split_fairness_11111111"
export PATH_PREFIX=/your/path/to/this/repo/InvDiff/data/

export LAMBDA=1
export DELTA_TYPE="small"
export SOFT_GROUPER=True
export PATH_GROUPER=t_g8_w0 # "t_g2_w3"
export LEARNING_RATE=1e-04
export BATCH_SIZE=64
export AC_STEPS=1
export LR_SCHEDULER="cosine"
export WARMUP_STEPS=1000
export DELTA_PARAM=0
export DELTA_RATIO=0.9
export HARD_GROUPER_NUM=8

text_num=2
sample_num=512

for checkpoint_step in 10000 #5000 3000
do
    python src/test.py \
        --path_prefix ${PATH_PREFIX} \
        --path_dataset ${DATASET_NAME} \
        --model_name "split_fairness_32111123-eiil-delta-lambda${LAMBDA}-delta${DELTA_RATIO}-${DELTA_TYPE}-deltaparam${DELTA_PARAM}-${SOFT_GROUPER}-${HARD_GROUPER_NUM}-${PATH_GROUPER}-bs${BATCH_SIZE}-sc${LEARNING_RATE}-ac${AC_STEPS}-lr${LR_SCHEDULER}-wu${WARMUP_STEPS}" \
        --model_checkpoint_num ${checkpoint_step} \
        --text_num ${text_num} \
        --sample_num ${sample_num} \
        --task_sample True \
        --save_sample True\
        --task_lpips False \
        --task_fid True \
        --task_recall True \
        --delta True \
        --lambda_value ${LAMBDA} \
        --delta_ratio ${DELTA_RATIO} \
        --delta_init0 False \
        --path_result_root invariant/results/fairness \
        --path_ckpt invariant/ckpts/fairness/models \
        --task_clip_score True \
        --task_bias_score True \
        --path_classifier invariant/ckpts/fairness/classifier/classifier_race/resnet18-64-0.0001 \
        --num_class_classifier 4
done
