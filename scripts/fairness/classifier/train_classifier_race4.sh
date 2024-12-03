
path_prefix="/your/path/InvDiff/data"
dataset_name="invariant/datasets/fairness/split_fairness_11111111"

batch_size=64
num_classes=4
num_epoch=10
model_name="resnet18"
learning_rate=0.0001
path_ckpt="invariant/ckpts/fairness/classifier/classifier_race"

# 运行 Python 脚本
python src/train_classifier.py \
    --path_prefix $path_prefix \
    --dataset_name $dataset_name \
    --batch_size $batch_size \
    --num_classes $num_classes \
    --num_epoch $num_epoch \
    --model_name $model_name \
    --learning_rate $learning_rate \
    --path_ckpt $path_ckpt \
    --label_name "Race" 