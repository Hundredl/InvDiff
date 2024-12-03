
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="invariant/datasets/fairness/split_fairness_32111123"
echo "start split_fairness_32111123 groupers group_num=16 loss_not_zero_weight=3"
python src/group/train_group.py     --path_embedding_file="invariant/ckpts/fairness/groupers/samples/split_fairness_32111123/samples_t.pkl"     --path_prefix="/your/path/to/this/repo/InvDiff/data/"     --path_groupers="invariant/ckpts/fairness/groupers/groupers/split_fairness_32111123"     --group_num=16     --loss_not_zero_weight=3 
echo "finish split_fairness_32111123 groupers"

