
import pickle
import torch
import torch.nn.functional as F
import argparse
import os

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_prefix', type=str, default='~/workspace/blob/drive/')
    parser.add_argument('--path_embedding_file', type=str, default='invariant/results/all_groupers/samples/samples_t.pkl')
    parser.add_argument('--path_groupers', type=str, default='invariant/results/all_groupers/groupers')
    parser.add_argument('--group_num', type=int, default=4)
    parser.add_argument('--loss_not_zero_weight', type=int, default=3)

    args = parser.parse_args()
    if 'AMLT_BLOB_DIR' in os.environ:
        # update the path prefix
        args.path_prefix = os.environ['AMLT_BLOB_DIR']
    print(args.path_prefix)
    args.path_embedding_file = os.path.join(args.path_prefix, args.path_embedding_file)
    args.path_groupers = os.path.join(args.path_prefix, args.path_groupers)

    return args

args = arg_parse()
embedding_file = args.path_embedding_file
with open(embedding_file, 'rb') as f:
    materials = pickle.load(f)

print(materials.keys()) # dict_keys(['samples', 'noise_targets', 'preds', 'indexs'])

samples = torch.cat(materials['samples'], dim=0)
noise_targets = torch.cat(materials['noise_targets'], dim=0)
preds = torch.cat(materials['preds'], dim=0)
indexs = torch.cat(materials['indexs'], dim=0)

group_num = args.group_num
loss_not_zero_weight = args.loss_not_zero_weight

sample_num = len(samples)


print(samples.shape, noise_targets.shape, preds.shape, indexs.shape)
print('sample_num:', sample_num)
print('group_num:', group_num)


loss = F.mse_loss(preds, samples, reduction='none')
loss = loss.mean(dim=list(range(1, len(loss.shape)))).reshape(-1, 1)

import torch

def optimize_weights(l, group_num, device='cuda'):
    sample_num = l.size(0)
    l = l.to(device)
    w = torch.rand((sample_num, group_num), requires_grad=True, device=device)
    print(f'w.shape: {w.shape}')  # 'w.shape: torch.Size([4795, 4])
    optimizer = torch.optim.Adam([w], lr=0.01)
    
    # to optimize the weights
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # calculate the loss for each group
        loss_per_group = l * w # (sample_num, group_num)
        
        sum_loss_per_group = torch.sum(loss_per_group, dim=0)
        count_per_group = torch.sum(w, dim=0)
        mean_loss_per_group = sum_loss_per_group / count_per_group # shape: (group_num,)

        # calculate the loss
        loss = -torch.std(mean_loss_per_group)
        loss-= loss_not_zero_weight * torch.min(loss_per_group)
        loss.backward()
        print(f'loss: {loss.item():.4f}')
        optimizer.step()
        
        # to make sure the weights are positive and sum to 1
        w.data = torch.max(w.data, torch.tensor(0.0))
        w.data = w.data / torch.sum(w.data, dim=1, keepdim=True)
    return w

weights = optimize_weights(loss, group_num)
print(f'weights.shape: {weights.shape}')
print(weights[:20])
print(weights.sum(dim=0))
print(f'start to save weights to {args.path_groupers}/weights_t_g{group_num}_w{loss_not_zero_weight}.pkl')
if not os.path.exists(args.path_groupers):
    os.makedirs(args.path_groupers)
with open(f'{args.path_groupers}/weights_t_g{group_num}_w{loss_not_zero_weight}.pkl', 'wb') as f:
    pickle.dump(weights, f)
print(f'weights saved to {args.path_groupers}/weights_t_g{group_num}_w{loss_not_zero_weight}.pkl')